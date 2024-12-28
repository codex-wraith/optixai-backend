from quart import Quart, jsonify, request, make_response, current_app
from quart_cors import cors
from moralis import evm_api
import aiohttp
import tempfile
import aiofiles
import base64
import requests
import replicate
import re
from redis.asyncio import Redis
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from web3 import Web3
from web3.exceptions import Web3Exception
from decimal import Decimal
from hypercorn.config import Config
from hypercorn.asyncio import serve
import google.generativeai as genai
import openai
import os
import asyncio
import logging
import uuid
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
INFURA_API_KEY = os.getenv('INFURA_API_KEY')
if not INFURA_API_KEY:
    raise EnvironmentError("INFURA_API_KEY not set in environment variables")
ZRX_API_KEY = os.getenv('ZRX_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MORALIS_API_KEY = os.getenv('MORALIS_API_KEY')
if not MORALIS_API_KEY:
    raise EnvironmentError("MORALIS_API_KEY not set in environment variables")
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
if not GOOGLE_API_KEY or not OPENAI_API_KEY:
    raise EnvironmentError("Missing API keys. Please set GOOGLE_API_KEY and OPENAI_API_KEY.")
if not STABILITY_API_KEY:
    raise EnvironmentError("STABILITY_API_KEY not set in environment variables")
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise EnvironmentError("REPLICATE_API_TOKEN not set in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)
client = openai.OpenAI(api_key=OPENAI_API_KEY)
GLOBAL_TRIAL_START_DATE = datetime(2024, 10, 6)
UNLIMITED_IMAGES = -1 
UNLIMITED_VIDEOS = -1 
TRIAL_IMAGE_COUNT = 50
TRIAL_VIDEO_COUNT = 25 
predictions = {}
WHITELISTED_ADDRESSES = [
    "0xe3dCD878B779C959A68fE982369E4c60c7503c38",  
    "0x780AfC062519614C83f1DbF9B320345772139e1e",
    "0xf52AfD0fF44aCfF80e9b3e54fe577E25af3f396E",
    "0xB48B371E4C6Af3ec298AdF6Dd32dec80a3Bffa09",
    "0x3fF749371f64526DCf706c10892663F374c61bD5"
]
SUBSCRIPTION_PLANS = {
    'Optix Core': {
        'images_per_month': 100,
        'videos_per_month': 0  # No video access for Tier 1
    },
    'Optix Blend': {
        'images_per_month': 500,
        'videos_per_month': 250
    },
    'Optix Pro': {
        'images_per_month': 1000,
        'videos_per_month': 500
    },
    'Optix Elite': {
        'images_per_month': 2000,
        'videos_per_month': 500
    }
}
PHASE_TIER_PERCENTAGES = {
    'Initial': {  # Under $1M
        'Optix Core': Decimal('2.5'),
        'Optix Blend': Decimal('3.0'),
        'Optix Pro':   Decimal('3.5'),
        'Optix Elite': Decimal('4.0')
    },
    'Growth': {   # $1M - $5M
        'Optix Core': Decimal('1.25'),
        'Optix Blend': Decimal('1.5'),
        'Optix Pro':   Decimal('1.75'),
        'Optix Elite': Decimal('2.0')
    },
    'Established': {  # Above $5M
        'Optix Core': Decimal('0.5'),
        'Optix Blend': Decimal('0.75'),
        'Optix Pro':   Decimal('1.0'),
        'Optix Elite': Decimal('1.25')
    }
}
app = Quart(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = Redis.from_url(os.environ.get('REDISCLOUD_URL'))
cors(app,
     allow_origin=['https://www.optixai.io', 'https://optixai.io', 'https://optixai.webflow.io', 'https://codex-wraith.github.io/optixaiui'],
     allow_methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'User-Address'],
     expose_headers=['Content-Type', 'User-Address'],
     allow_credentials=True)

@app.route('/price', methods=['GET'])
async def get_price():
    # Extract parameters from the request
    chain_id = request.args.get('chainId')
    sell_token = request.args.get('sellToken')
    buy_token = request.args.get('buyToken')
    sell_amount = request.args.get('sellAmount')
    taker = request.args.get('taker')
    slippage_bps = request.args.get('slippageBps', '200')  # Default to 2% slippage

    # Construct the 0x API URL
    base_url = "https://api.0x.org"
    endpoint = f"/swap/permit2/price"
    url = f"{base_url}{endpoint}"

    # Prepare the query parameters
    params = {
        "chainId": chain_id,
        "sellToken": sell_token,
        "buyToken": buy_token,
        "sellAmount": sell_amount,
        "taker": taker,
        "slippageBps": slippage_bps
    }

    # Prepare headers
    headers = {
        "0x-api-key": ZRX_API_KEY,
        "0x-version": "v2"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return jsonify(data)
            else:
                error_data = await response.json()
                return jsonify(error_data), response.status

@app.route('/quote', methods=['GET'])
async def get_quote():
    # Extract parameters from the request
    chain_id = request.args.get('chainId')
    sell_token = request.args.get('sellToken')
    buy_token = request.args.get('buyToken')
    sell_amount = request.args.get('sellAmount')
    taker = request.args.get('taker')
    slippage_bps = request.args.get('slippageBps', '200')  # Default to 2% slippage

    # Construct the 0x API URL
    base_url = "https://api.0x.org"
    endpoint = f"/swap/permit2/quote"
    url = f"{base_url}{endpoint}"

    # Prepare the query parameters
    params = {
        "chainId": chain_id,
        "sellToken": sell_token,
        "buyToken": buy_token,
        "sellAmount": sell_amount,
        "taker": taker,
        "slippageBps": slippage_bps
    }

    # Prepare headers
    headers = {
        "0x-api-key": ZRX_API_KEY,
        "0x-version": "v2"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return jsonify(data)
            else:
                error_data = await response.json()
                return jsonify(error_data), response.status


@app.route('/description', methods=['GET', 'OPTIONS'])
async def get_description():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    description = (
        "Connect your wallet to access OptixAI's advanced image and video generation. "
        "Create stunning visuals with our AI tools, and unlock video generation to bring your creations to life."
    )
    return jsonify({'description': description})


@app.route('/tiers-info', methods=['GET', 'OPTIONS'])
async def tiers_info():
    if request.method == 'OPTIONS':
        return await handle_options_request()
    
    try:
        # Get token supply info
        contract = get_token_contract()
        total_supply = contract.functions.totalSupply().call()
        token_decimals = contract.functions.decimals().call()
        total_supply_adjusted = Decimal(total_supply) / Decimal(10 ** token_decimals)
        
        # Get current market cap from cache
        cap_str = await app.redis_client.get("moralis_market_cap")
        if not cap_str:
            # If cache missing, trigger update
            await update_moralis_price_job()
            cap_str = await app.redis_client.get("moralis_market_cap")
        
        market_cap = Decimal(cap_str)
        current_phase = get_current_phase(market_cap)
        current_percentages = PHASE_TIER_PERCENTAGES[current_phase]
        
        tiers_info = {}
        for tier, plan in SUBSCRIPTION_PLANS.items():
            percentage = current_percentages[tier]  # Get dynamic percentage based on phase
            tokens_required = (percentage / Decimal(100)) * total_supply_adjusted
            tiers_info[tier] = {
                'percentage': float(percentage),
                'tokensRequired': float(tokens_required),
                'imagesPerMonth': plan['images_per_month'],
                'videosPerMonth': plan['videos_per_month'],
                'hasVideoAccess': plan['videos_per_month'] > 0,
                'isUltraTier': tier == 'Optix Elite',
                'features': {
                    'imageGeneration': True,
                    'videoGeneration': plan['videos_per_month'] > 0,
                    'ultraQuality': tier == 'Optix Elite'
                }
            }
        
        return jsonify({
            'success': True,
            'totalSupply': float(total_supply_adjusted),
            'tiers': tiers_info,
            'videoTierMinimum': 'Optix Blend',  # Indicates minimum tier for video access
            'currentPhase': current_phase  # Optionally include phase info
        })
    except Exception as e:
        logging.error(f"Error fetching tiers info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/check-whitelist', methods=['GET', 'OPTIONS'])
async def check_whitelist():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    user_address = request.headers.get('User-Address')
    if not user_address:
        return jsonify({'error': 'User address not provided'}), 400

    is_whitelisted_user = is_whitelisted(user_address)

    return jsonify({
        'isWhitelisted': is_whitelisted_user
    })

@app.route('/image-ai')
async def proxy_image():
    file_id = request.args.get('id')
    if not file_id:
        return await make_response('Image ID is required', 400)

    file_path = await app.redis_client.get(f"image_{file_id}")
    if not file_path:
        return await make_response('Image not found', 404)

    if not os.path.exists(file_path):
        await app.redis_client.delete(f"image_{file_id}")
        return await make_response('Image file not found', 404)

    try:
        async with aiofiles.open(file_path, 'rb') as file:
            image_data = await file.read()
        
        # Don't delete the file immediately, set a longer expiration in Redis instead
        await app.redis_client.expire(f"image_{file_id}", 3600)  # Expire after 1 hour
        
        proxy_response = await make_response(image_data)
        proxy_response.headers['Content-Type'] = 'image/png'
        proxy_response.headers['Access-Control-Allow-Origin'] = '*'  # Or specify a single origin
        return proxy_response
    except Exception as e:
        current_app.logger.error(f"Error serving image: {str(e)}")
        return await make_response(f'Error serving image: {str(e)}', 500)


@app.route('/video-ai')
async def proxy_video():
    prediction_id = request.args.get('id')
    if not prediction_id:
        return await make_response('Video ID is required', 400)

    prediction = predictions.get(prediction_id)
    if not prediction or 'output' not in prediction:
        return await make_response('Video not found', 404)

    try:
        # Get the original video URL
        video_url = prediction['output']
        
        # Download the video
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    return await make_response('Failed to fetch video', 500)
                
                video_data = await response.read()

        # Create response with proper headers
        proxy_response = await make_response(video_data)
        proxy_response.headers['Content-Type'] = 'video/mp4'
        proxy_response.headers['Access-Control-Allow-Origin'] = '*'
        return proxy_response

    except Exception as e:
        current_app.logger.error(f"Error serving video: {str(e)}")
        return await make_response(f'Error serving video: {str(e)}', 500)


@app.route('/upload-image', methods=['POST', 'OPTIONS'])
async def upload_image():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    try:
        files = await request.files
        if 'image' not in files:
            return jsonify({'error': 'No image file provided'}), 400

        file = files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Create a unique filename
        file_id = str(uuid.uuid4())
        temp_path = f"/tmp/{file_id}.png"

        # Save the file
        await file.save(temp_path)

        try:
            # Upload to tmpfiles.org
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('file',
                             open(temp_path, 'rb'),
                             filename='image.png',
                             content_type='image/png')

                async with session.post('https://tmpfiles.org/api/v1/upload', data=data) as response:
                    if response.status != 200:
                        raise Exception('Failed to upload image')
                    
                    result = await response.json()
                    # The tmpfiles.org response structure is different
                    # It returns something like: {"status":"ok","data":{"url":"https://tmpfiles.org/1234/image.png"}}
                    if not result.get('data', {}).get('url'):
                        raise Exception('Invalid response from tmpfiles.org')
                    
                    # Convert the URL to a direct download link
                    tmp_url = result['data']['url']
                    direct_url = tmp_url.replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')
                    
                    app.logger.info(f"Uploaded image. Original URL: {tmp_url}, Direct URL: {direct_url}")

            # Store the URL in Redis for later use
            await app.redis_client.set(f"video_image_{file_id}", direct_url, ex=3600)  # Expire after 1 hour

            return jsonify({
                'success': True,
                'image_url': direct_url,
                'file_id': file_id
            })

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        app.logger.error(f"Error in upload_image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/user-session', methods=['POST', 'GET', 'OPTIONS'])
async def user_session():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    user_address = request.headers.get('User-Address')
    if not user_address:
        return jsonify({'error': 'User address not provided'}), 400

    if is_whitelisted(user_address):
        return jsonify(
            success=True,
            freeTrialActive=False,
            trialTimeLeft=0,
            imagesLeft=UNLIMITED_IMAGES,
            videosLeft=UNLIMITED_VIDEOS,
            videoLimit=UNLIMITED_VIDEOS,
            tier='Unlimited',
            availableUpgrades=[],
            hasVideoAccess=True,
            hasUltraAccess=True
        )

    # Ensure Redis client is initialized
    if not app.redis_client:
        app.redis_client = await create_redis_client()

    user_prefix = f"user_{user_address}_"
    free_trial_active, remaining_time, trial_image_count, trial_video_count = await is_free_trial_active(user_address)

    if request.method == 'POST':
        images_left, videos_left, tier_status = await get_or_initialize_user_data(
            user_prefix, free_trial_active, user_address
        )
        if images_left is None:
            return jsonify({'error': 'No images left to use'}), 400

        actual_tier_status, meets_requirements = await verify_tier_for_user(user_address)

        if actual_tier_status != 'None' and meets_requirements and free_trial_active:
            await app.redis_client.set(f"user_{user_address}_free_trial_override", 'False')
            free_trial_active = False
            tier_status = actual_tier_status
        elif free_trial_active:
            tier_status = 'Free Trial'
            if not await app.redis_client.exists(f"{user_prefix}images_left"):
                await app.redis_client.set(f"{user_prefix}images_left", str(trial_image_count))
            if not await app.redis_client.exists(f"{user_prefix}videos_left"):
                await app.redis_client.set(f"{user_prefix}videos_left", str(trial_video_count))
            images_left = int(await app.redis_client.get(f"{user_prefix}images_left") or trial_image_count)
            videos_left = int(await app.redis_client.get(f"{user_prefix}videos_left") or trial_video_count)
        else:
            tier_status = actual_tier_status

        await app.redis_client.set(f"{user_prefix}tier", tier_status)

    else:
        # GET request handling
        actual_tier_status, meets_requirements = await verify_tier_for_user(user_address)
        tier_status = await app.redis_client.get(f"{user_prefix}tier") or 'None'

        if tier_status != actual_tier_status:
            # Tier has changed, update counts
            images_left, videos_left, tier_status = await get_or_initialize_user_data(
                user_prefix, free_trial_active, user_address
            )
        else:
            if free_trial_active:
                if tier_status == 'None' or tier_status == 'Free Trial':
                    tier_status = 'Free Trial'
                    # Initialize counts if they don't exist
                    if not await app.redis_client.exists(f"{user_prefix}images_left"):
                        await app.redis_client.set(f"{user_prefix}images_left", str(trial_image_count))
                    if not await app.redis_client.exists(f"{user_prefix}videos_left"):
                        await app.redis_client.set(f"{user_prefix}videos_left", str(trial_video_count))

                # Always get the current counts from Redis
                images_left = int(await app.redis_client.get(f"{user_prefix}images_left") or trial_image_count)
                videos_left = int(await app.redis_client.get(f"{user_prefix}videos_left") or trial_video_count)
            else:
                # Non-trial user
                images_left = int(await app.redis_client.get(f"{user_prefix}images_left") or 0)
                videos_left = int(await app.redis_client.get(f"{user_prefix}videos_left") or 0)

    # Update available upgrades logic
    all_tiers = ['Optix Core', 'Optix Blend', 'Optix Pro', 'Optix Elite']
    if tier_status == 'Free Trial':
        available_upgrades = all_tiers
    else:
        current_tiers = tier_status.split(', ')
        available_upgrades = [tier for tier in all_tiers if tier not in current_tiers]

    # Add Ultra access flag
    has_ultra_access = tier_status == 'Optix Elite' or is_whitelisted(user_address)

    # Determine video access based on tier
    has_video_access = (
        is_whitelisted(user_address) or
        free_trial_active or
        tier_status in ['Optix Blend', 'Optix Pro', 'Optix Elite']
    )

    # Get video limit based on tier
    if tier_status in SUBSCRIPTION_PLANS:
        video_limit = SUBSCRIPTION_PLANS[tier_status]['videos_per_month']
    elif free_trial_active:
        video_limit = trial_video_count
    else:
        video_limit = 0

    return jsonify(
        success=True,
        freeTrialActive=free_trial_active,
        trialTimeLeft=remaining_time,
        imagesLeft=images_left,
        videosLeft=videos_left,
        videoLimit=video_limit,
        tier=tier_status,
        availableUpgrades=available_upgrades,
        hasUltraAccess=has_ultra_access,
        hasVideoAccess=has_video_access
    )



@app.route('/trial-status', methods=['GET', 'OPTIONS'])
async def get_trial_status():
    if request.method == 'OPTIONS':
        return '', 204

    user_address = request.headers.get('User-Address')
    free_trial_active, remaining_time, _, _ = await is_free_trial_active(user_address)

    return jsonify({
        'freeTrialActive': free_trial_active,
        'trialTimeLeft': remaining_time
    })


@app.route('/generate-video', methods=['POST', 'OPTIONS'])
async def generate_video():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    try:
        data = await request.get_json()
        user_address = request.headers.get('User-Address') or data.get('userAddress')
        prompt = data.get('prompt')
        first_frame_image = data.get('first_frame_image')
        prompt_optimizer = data.get('prompt_optimizer', True)

        if not user_address:
            return jsonify({'error': 'User address not provided'}), 400

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Check whitelist status and initialize variables
        is_whitelisted_user = is_whitelisted(user_address)
        videos_left = UNLIMITED_VIDEOS
        tier_status = 'Unlimited' if is_whitelisted_user else None

        if not is_whitelisted_user:
            user_prefix = f"user_{user_address}_"
            free_trial_active, _, _, _ = await is_free_trial_active(user_address)
            tier_status = await get_user_tier_status(user_address)

            # Check if user has video access based on tier
            has_video_access = (
                tier_status in ['Optix Blend', 'Optix Pro', 'Optix Elite'] or 
                free_trial_active
            )

            if not has_video_access:
                return jsonify({
                    'error': 'Video generation requires Tier 2 or higher subscription',
                    'tier': tier_status
                }), 403

            # Get current video count without decrementing
            _, videos_left, current_tier = await get_or_initialize_user_data(
                user_prefix, 
                free_trial_active, 
                user_address, 
                decrement_images=False,
                decrement_videos=False
            )

            # Check if videos left
            if videos_left <= 0:
                return jsonify({
                    'error': 'No video generations left',
                    'videosLeft': 0,
                    'tier': current_tier
                }), 400

            # Decrement video count if not whitelisted
            try:
                _, videos_left, tier_status = await get_or_initialize_user_data(
                    user_prefix,
                    free_trial_active,
                    user_address,
                    decrement_images=False,
                    decrement_videos=True
                )
            except ValueError as e:
                return jsonify({
                    'error': str(e),
                    'tier': tier_status
                }), 403

        # Log the generation attempt
        app.logger.info(f"Video generation initiated - User: {user_address}, Tier: {tier_status}, Videos Left: {videos_left}")

        # Generate unique ID for this prediction
        prediction_id = str(uuid.uuid4())

        # Start prediction in background task
        asyncio.create_task(run_prediction(
            prediction_id,
            prompt,
            first_frame_image,
            prompt_optimizer
        ))

        # Return response with comprehensive data
        response_data = {
            'success': True,
            'prediction_id': prediction_id,
            'videosLeft': videos_left,
            'tier': tier_status,
            'isWhitelisted': is_whitelisted_user,
            'freeTrialActive': free_trial_active if not is_whitelisted_user else False
        }

        if not is_whitelisted_user and tier_status in SUBSCRIPTION_PLANS:
            response_data['videoLimit'] = SUBSCRIPTION_PLANS[tier_status]['videos_per_month']

        app.logger.info(f"Video generation request successful - ID: {prediction_id}")
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error starting video generation: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500



@app.route('/video-progress/<prediction_id>', methods=['GET', 'OPTIONS'])
async def get_video_progress(prediction_id):
    if request.method == 'OPTIONS':
        return await handle_options_request()

    try:
        if prediction_id not in predictions:
            response = await make_response(jsonify({'error': 'Prediction not found'}), 404)
            response.headers['Access-Control-Allow-Origin'] = 'https://www.optixai.io, https://optixai.io, https://optixai.webflow.io, https://codex-wraith.github.io/optixaiui'
            return response

        prediction = predictions[prediction_id]
        
        try:
            # Get the latest prediction status from Replicate
            replicate_prediction = replicate.predictions.get(prediction.get('replicate_id'))
            
            # Update our local prediction status based on Replicate's status
            if replicate_prediction.status == 'succeeded':
                prediction.update({
                    'status': 'succeeded',
                    'progress': 100,
                    'output': replicate_prediction.output  # Direct URL from Replicate
                })
            elif replicate_prediction.status == 'failed':
                prediction.update({
                    'status': 'failed',
                    'error': replicate_prediction.error or 'Video generation failed',
                    'progress': 0
                })
            elif replicate_prediction.status == 'processing':
                prediction.update({
                    'status': 'processing',
                    'progress': 50
                })
            elif replicate_prediction.status == 'starting':
                prediction.update({
                    'status': 'processing',
                    'progress': 25
                })
        except Exception as e:
            app.logger.error(f"Error fetching Replicate prediction: {str(e)}")
            # Continue with existing prediction data if there's an error

        # Clean up completed predictions after some time
        if prediction['status'] in ['succeeded', 'failed']:
            asyncio.create_task(cleanup_prediction(prediction_id))

        response_data = {
            'status': prediction['status'],
            'progress': prediction['progress'],
            'output': prediction.get('output'),  # Direct URL from Replicate
            'error': prediction.get('error')
        }

        response = await make_response(jsonify(response_data))
        response.headers['Access-Control-Allow-Origin'] = 'https://www.optixai.io, https://optixai.io, https://optixai.webflow.io, https://codex-wraith.github.io/optixaiui'
        return response

    except Exception as e:
        app.logger.error(f"Error in get_video_progress: {str(e)}")
        error_response = await make_response(
            jsonify({
                'error': str(e),
                'status': 'failed',
                'progress': 0
            }), 
            503
        )
        error_response.headers['Access-Control-Allow-Origin'] = 'https://www.optixai.io, https://optixai.io, https://optixai.webflow.io, https://codex-wraith.github.io/optixaiui'
        return error_response



@app.route('/generate', methods=['POST', 'OPTIONS'])
async def generate_content():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    try:
        data = await request.get_json()
        user_address = data.get('userAddress')
        
        if not user_address:
            return jsonify({'error': 'User address is required'}), 400

        # Check whitelist status first
        is_whitelisted_user = is_whitelisted(user_address)
        
        # Extract aspect ratio, default to '1:1' if not provided
        aspect_ratio = data.get('aspectRatio', '1:1')
        aspect_ratio_mapping = {
            '1:1': '1:1',
            '16:9': '16:9',
            '9:16': '9:16',
            '3:2': '3:2',
            '2:3': '2:3',
            '5:4': '5:4',
        }
        selected_aspect_ratio = aspect_ratio_mapping.get(aspect_ratio, '1:1')
    
        # Get trial status and tier status
        free_trial_active, _, _, _ = await is_free_trial_active(user_address)
        tier_status = await get_user_tier_status(user_address)

        app.logger.info(f"Generate request - User: {user_address}, Whitelisted: {is_whitelisted_user}, Trial: {free_trial_active}, Tier: {tier_status}")

        # Determine which tier and prompt to use
        prompt_used = None
        model_version = "black-forest-labs/flux-1.1-pro"

        # Modified tier checks to properly handle whitelisted users
        if is_whitelisted_user:
            # Whitelisted users can use any tier
            if data.get('tier4Prompt'):
                prompt_used = data['tier4Prompt']
                tier_used = 'Optix Elite'
                model_version = "black-forest-labs/flux-1.1-pro-ultra"
            elif data.get('tier3Prompt'):
                prompt_used = data['tier3Prompt']
                tier_used = 'Optix Pro'
            elif data.get('tier2Prompt'):
                prompt_used = data['tier2Prompt']
                tier_used = 'Optix Blend'
            elif data.get('tier1Prompt'):
                prompt_used = data['tier1Prompt']
                tier_used = 'Optix Core'
        else:
            # Non-whitelisted users follow regular tier logic
            if data.get('tier4Prompt') and (tier_status == 'Optix Elite' or free_trial_active):
                prompt_used = data['tier4Prompt']
                tier_used = 'Optix Elite'
                model_version = "black-forest-labs/flux-1.1-pro-ultra"
            elif data.get('tier3Prompt') and (free_trial_active or tier_status in ['Optix Pro', 'Optix Elite']):
                prompt_used = data['tier3Prompt']
                tier_used = 'Optix Pro'
            elif data.get('tier2Prompt') and (free_trial_active or tier_status in ['Optix Blend', 'Optix Pro', 'Optix Elite']):
                prompt_used = data['tier2Prompt']
                tier_used = 'Optix Blend'
            elif data.get('tier1Prompt') and (free_trial_active or tier_status in [ 'Optix Core', 'Optix Blend', 'Optix Pro', 'Optix Elite']):
                prompt_used = data['tier1Prompt']
                tier_used = 'Optix Core'

        if not prompt_used:
            app.logger.error(f"No valid prompt - User: {user_address}, Prompts received: {[k for k in data.keys() if 'Prompt' in k]}")
            return jsonify({'error': 'No valid prompt provided or unauthorized tier access'}), 400

        logging.info(f"Original prompt ({tier_used}): {prompt_used}")

        prompts = {
            'Optix Core': f"""Refine this prompt for a pixel art style image with cartoon elements: '{prompt_used}'
            Create a single, detailed prompt that:
            - Emphasizes classic pixel art aesthetics and charm
            - Incorporates playful cartoon-style elements
            - Key compositional elements (foreground, background)
            - Suggests color palette and mood
            Avoid:
            - Technical instructions (e.g., resolution numbers, color codes)
            - Mentions of UI elements, borders, or non-artistic components
            - Realistic or photographic elements
            Present the output as one comprehensive, descriptive sentence that naturally blends pixel art and cartoon elements.""",

            'Optix Blend': f"""Refine this prompt for an image blending pixel art (40%) with photorealistic (60%) elements: '{prompt_used}'
            Create a single, detailed prompt that:
            - Specifies elements to be rendered in pixel art style
            - Describes photorealistic details
            - Includes key compositional elements (foreground, background)
            - Suggests lighting and textures
            Avoid:
            - Mentioning percentages or ratios
            - Technical instructions about pixel structures
            - References to UI elements or non-artistic components
            Present the output as one comprehensive, descriptive sentence without separating pixel and realistic elements.""",

            'Optix Pro': f"""Enhance this prompt for a photorealistic image with subtle artistic touches: '{prompt_used}'
            Craft a single, detailed prompt that includes:
            - Specific lighting details (direction, intensity, color)
            - Subtle artistic textures that maintain realism
            - Key compositional elements (foreground, background)
            - Mood and atmosphere
            Ensure to:
            - Keep the original intent and key elements of the prompt
            - Avoid mentioning non-photorealistic or obvious digital effects
            Output a single, comprehensive sentence that guides the AI to create a highly detailed, photorealistic image with artistic nuances.""",

            'Optix Elite': f"""Enhance this prompt for an ultra-realistic image with exceptional detail and artistic mastery: '{prompt_used}'
            Craft a single, detailed prompt that includes:
            - Photorealistic lighting and atmospheric conditions
            - Intricate textures and surface details
            - Complex compositional elements
            - Advanced depth and perspective
            - Sophisticated color grading
            Ensure to:
            - Maintain hyperrealistic quality
            - Include cinematic elements
            - Preserve artistic nuances
            Output a single, comprehensive sentence that guides the AI to create an ultra-detailed, masterfully composed image."""
        }

        refined_prompt = prompts[tier_used]
        # Generate text based on the refined prompt
        response = await generate_text(refined_prompt)

        # Post-process the refined prompt
        def post_process_prompt(refined_prompt):
            # Remove any instructions that might have slipped through
            refined_prompt = re.sub(r'(?i)(create|generate|produce|make)( an? (image|picture|artwork|illustration))?( of)?:?\s*', '', refined_prompt)
            
            # Ensure it's a single sentence
            refined_prompt = refined_prompt.split('.')[0].strip()
            
            # Add any prefixes or suffixes your image AI responds well to
            refined_prompt = f"Create a detailed image of {refined_prompt}, high quality, intricate details"
            
            return refined_prompt

        final_prompt = post_process_prompt(response.text)
        logging.info(f"Final prompt: {final_prompt}")

        # Use appropriate model based on tier
        if model_version == "black-forest-labs/flux-1.1-pro-ultra":
            # Ultra model parameters
            model_params = {
                "prompt": final_prompt,
                "raw": True,
                "image_prompt_strength": 0.8,
                "aspect_ratio": selected_aspect_ratio,
                "output_format": "png",
                "output_quality": 100,
                "safety_tolerance": 6
            }
        else:
            # Standard model parameters
            model_params = {
                "prompt": final_prompt,
                "prompt_upsampling": False,
                "aspect_ratio": selected_aspect_ratio,
                "output_format": "png",
                "output_quality": 100,
                "safety_tolerance": 6
            }

        output = replicate.run(
            model_version,
            input=model_params,
            api_token=REPLICATE_API_TOKEN
        )

        # The output is directly a URL
        image_url = output
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_data = base64.b64encode(image_response.content).decode('utf-8')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(base64.b64decode(image_data))
            temp_file_path = temp_file.name

        file_id = str(uuid.uuid4())  # Generate a unique ID
        await app.redis_client.set(f"image_{file_id}", temp_file_path, ex=1800)  # Store for 30 minutes
        
                # Adjust the user's image count post-generation using their known user_address
        is_whitelisted_user = is_whitelisted(user_address)
        if not is_whitelisted_user:
            user_prefix = f"user_{user_address}_"
            free_trial_active, _, trial_images, trial_videos = await is_free_trial_active(user_address)

            if free_trial_active:
                # Get current counts from Redis
                current_images = int(await app.redis_client.get(f"{user_prefix}images_left") or trial_images)
                current_videos = int(await app.redis_client.get(f"{user_prefix}videos_left") or trial_videos)
                
                # Check if there are images left
                if current_images <= 0:
                    return jsonify({
                        'error': 'No trial images left to use',
                        'imagesLeft': 0,
                        'tier': 'Free Trial',
                        'freeTrialActive': True
                    }), 400

                # Decrement and update image count
                current_images -= 1
                await app.redis_client.set(f"{user_prefix}images_left", str(current_images))
                
                return jsonify({
                    'text': final_prompt,
                    'file_id': file_id,
                    'imagesLeft': current_images,
                    'videosLeft': current_videos,
                    'tier': 'Free Trial',
                    'freeTrialActive': True
                })
            else:
                # Not in trial, use regular subscription logic
                # First, get the current image count without decrementing
                images_left, videos_left, current_tier = await get_or_initialize_user_data(
                    user_prefix, 
                    free_trial_active, 
                    user_address, 
                    decrement_images=False,
                    decrement_videos=False
                )

                # Check if there are images left
                if images_left <= 0:
                    return jsonify({
                        'error': 'No images left to use',
                        'imagesLeft': 0,
                        'tier': current_tier
                    }), 400

                # If we have images, then decrement and get the new count
                images_left, videos_left, current_tier = await get_or_initialize_user_data(
                    user_prefix, 
                    free_trial_active, 
                    user_address, 
                    decrement_images=True,
                    decrement_videos=False
                )

                return jsonify({
                    'text': final_prompt,
                    'file_id': file_id,
                    'imagesLeft': images_left,
                    'videosLeft': videos_left,
                    'tier': current_tier,
                    'freeTrialActive': False
                })
        else:
            return jsonify({
                'text': final_prompt,
                'file_id': file_id,
                'imagesLeft': UNLIMITED_IMAGES,
                'videosLeft': UNLIMITED_VIDEOS,
                'tier': 'Unlimited',
                'freeTrialActive': False
            })

    except Exception as e:
        app.logger.error(f"Error generating content: {str(e)}")
        return jsonify({'error': str(e)}), 500


async def verify_tier_for_user(user_address):
    """
    Determines which tier the user qualifies for, using the cached Moralis price 
    and dynamic PHASE_TIER_PERCENTAGES. 
    """
    if is_whitelisted(user_address):
        return 'Unlimited', True

    try:
        # 1) Get the user’s % of total supply
        checksum_address = Web3.to_checksum_address(user_address)
        percentage_held, total_supply_adjusted = verify_user_holdings(checksum_address)

        # 2) Read cached price & market cap from Redis
        price_str = await app.redis_client.get("moralis_usd_price")
        cap_str = await app.redis_client.get("moralis_market_cap")

        # If absent, you can fetch directly OR treat user as 'None'
        if not price_str or not cap_str:
            logging.warning("[verify_tier_for_user] No cached Moralis price found, fallback to direct fetch.")
            # (A) Fallback to direct fetch:
            usd_price_per_token = get_token_price_from_moralis()
            market_cap = total_supply_adjusted * usd_price_per_token
        else:
            # Use the cached values
            usd_price_per_token = Decimal(price_str)
            market_cap = Decimal(cap_str)

        # 3) Determine the current phase (Initial, Growth, Established)
        current_phase = get_current_phase(market_cap)

        # 4) Get dynamic percentages for that phase
        dynamic_percentages = PHASE_TIER_PERCENTAGES[current_phase]

        # 5) Sort them in descending order 
        #    so we check highest required % (Elite) first, etc.
        sorted_tiers = sorted(
            dynamic_percentages.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 6) Compare user’s actual holding % 
        for tier_name, required_pct in sorted_tiers:
            if percentage_held >= required_pct:
                return tier_name, True

        return 'None', False

    except Exception as e:
        logging.error(f"Error verifying tier for user {user_address}: {str(e)}")
        return 'None', False

def get_web3_instance():
    providers = [
        Web3.HTTPProvider(f'https://base-mainnet.infura.io/v3/{INFURA_API_KEY}'),
        # You can add backup providers here, e.g.:
        # Web3.HTTPProvider('https://eth-mainnet.alchemyapi.io/v2/YOUR-ALCHEMY-KEY'),
    ]

    for provider in providers:
        try:
            web3 = Web3(provider)
            if web3.is_connected():
                return web3
        except Web3Exception as e:
            logging.error(f"Failed to connect to {provider.endpoint_uri}: {e}")
    
    raise ConnectionError("Failed to connect to any Ethereum provider")

def get_token_contract():
    token_address = "0x0464a6939B0e341Ed502B2c4a6Dc1e60884762DF"
    token_abi = [
        {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
         "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
         "type": "function"},
        {"constant": True, "inputs": [],
         "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}],
         "type": "function"},
        {"constant": True, "inputs": [],
         "name": "decimals", "outputs": [{"name": "", "type": "uint8"}],
         "type": "function"}
    ]
    web3 = get_web3_instance()
    checksum_token_address = Web3.to_checksum_address(token_address)
    return web3.eth.contract(address=checksum_token_address, abi=token_abi)

def verify_user_holdings(checksum_address):
    try:
        contract = get_token_contract()
        
        balance = contract.functions.balanceOf(checksum_address).call()
        total_supply = contract.functions.totalSupply().call()
        token_decimals = contract.functions.decimals().call()
        
        # Adjust balance and total supply for decimals
        balance_adjusted = Decimal(balance) / Decimal(10 ** token_decimals)
        total_supply_adjusted = Decimal(total_supply) / Decimal(10 ** token_decimals)
        
        if total_supply_adjusted == 0:
            logging.error("Total supply is zero, cannot calculate percentage")
            return Decimal(0), total_supply_adjusted
        
        percentage_held = (balance_adjusted / total_supply_adjusted) * 100
        return percentage_held, total_supply_adjusted
    except Exception as e:
        logging.error(f"Error in retrieving token holdings for {checksum_address}: {str(e)}")
        return Decimal(0), Decimal(0)

def is_whitelisted(address):
    return address.lower() in (addr.lower() for addr in WHITELISTED_ADDRESSES)

def get_token_price_from_moralis() -> Decimal:
    params = {
        "chain": "base",
        "address": "0x0464a6939B0e341Ed502B2c4a6Dc1e60884762DF"
    }

    try:
        result = evm_api.token.get_token_price(
            api_key=MORALIS_API_KEY,
            params=params,
        )
        # The Moralis response has "usdPrice"
        usd_price_str = str(result["usdPrice"])
        return Decimal(usd_price_str)
    except Exception as e:
        logging.error(f"Error fetching price from Moralis: {e}")
        # fallback to 0 or raise
        return Decimal(0)

def get_current_phase(market_cap: Decimal) -> str:
    if market_cap < Decimal('1000000'):  # Under $1M
        return 'Initial'
    elif market_cap <= Decimal('5000000'):  # $1M - $5M
        return 'Growth'
    else:
        return 'Established'


async def run_prediction(prediction_id, prompt, first_frame_image, prompt_optimizer):
    try:
        # Initialize prediction status
        predictions[prediction_id] = {
            'status': 'starting',
            'progress': 0,
            'output': None,
            'error': None,
            'start_time': datetime.now()
        }

        if not prompt:
            raise ValueError("Prompt is required")

        # Initialize input parameters
        input_params = {
            "prompt": prompt,
            "prompt_optimizer": prompt_optimizer
        }

        # Only process and include first_frame_image if it's provided
        if first_frame_image:
            # Process image upload if needed
            if first_frame_image.startswith('data:') or first_frame_image.startswith('blob:'):
                async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    if first_frame_image.startswith('data:'):
                        image_data = base64.b64decode(first_frame_image.split(',')[1])
                        await aiofiles.os.write(temp_file.fileno(), image_data)
                    else:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(first_frame_image) as response:
                                image_data = await response.read()
                                await aiofiles.os.write(temp_file.fileno(), image_data)
                    
                    image_path = temp_file.name

                try:
                    # Upload to tmpfiles.org
                    async with aiohttp.ClientSession() as session:
                        data = aiohttp.FormData()
                        data.add_field('file',
                                     open(image_path, 'rb'),
                                     filename='image.png',
                                     content_type='image/png')
                        
                        async with session.post('https://tmpfiles.org/api/v1/upload',
                                              data=data) as response:
                            if response.status != 200:
                                raise Exception("Failed to upload image")
                            
                            result = await response.json()
                            tmp_url = result['data']['url']
                            image_url = tmp_url.replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')
                finally:
                    if os.path.exists(image_path):
                        await aiofiles.os.remove(image_path)
            else:
                image_url = first_frame_image

            # Add image URL to input parameters only if we have one
            input_params["first_frame_image"] = image_url

        app.logger.info(f"Starting video generation with prompt: {prompt}")

        # Create prediction using async method
        replicate_prediction = await replicate.predictions.async_create(
            model="minimax/video-01",
            input=input_params
        )

        # Store the Replicate prediction ID and start monitoring
        predictions[prediction_id].update({
            'replicate_id': replicate_prediction.id,
            'status': 'processing',
            'progress': 0
        })

        app.logger.info(f"Created prediction with ID: {replicate_prediction.id}")

        # Monitor prediction progress
        while True:
            try:
                # Get latest prediction status
                current_prediction = replicate.predictions.get(replicate_prediction.id)
                elapsed_time = (datetime.now() - predictions[prediction_id]['start_time']).total_seconds()
                
                if current_prediction.status == 'succeeded':
                    predictions[prediction_id].update({
                        'status': 'succeeded',
                        'progress': 100,
                        'output': current_prediction.output
                    })
                    break
                elif current_prediction.status == 'failed':
                    predictions[prediction_id].update({
                        'status': 'failed',
                        'progress': 0,
                        'error': current_prediction.error or 'Video generation failed'
                    })
                    break
                else:
                    # Calculate progress based on elapsed time
                    progress = min(int((elapsed_time / 60) * 100), 99)  # Cap at 99% until complete
                    predictions[prediction_id].update({
                        'status': current_prediction.status,
                        'progress': progress
                    })

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                app.logger.error(f"Error checking prediction status: {str(e)}")
                await asyncio.sleep(2)

    except Exception as e:
        app.logger.error(f"Error in video generation: {str(e)}")
        predictions[prediction_id].update({
            'status': 'failed',
            'error': str(e)
        })


async def cleanup_prediction(prediction_id):
    await asyncio.sleep(3600)  # Keep prediction data for 1 hour instead of 5 minutes
    predictions.pop(prediction_id, None)

async def generate_text(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    # Log the refined prompt
    logging.info(f"Refined prompt: {response.text}")
    
    return response

async def handle_options_request():
    """Handle OPTIONS request for CORS."""
    response = await make_response('', 204)
    response.headers['Access-Control-Allow-Origin'] = 'https://www.optixai.io, https://optixai.io, https://optixai.webflow.io, https://codex-wraith.github.io/optixaiui'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, User-Address'
    return response

async def get_user_tier_status(user_address):
    """Helper function to get user's current tier status."""
    if is_whitelisted(user_address):
        return 'Unlimited'
    
    user_prefix = f"user_{user_address}_"
    tier_status = await app.redis_client.get(f"{user_prefix}tier")
    return tier_status or 'None'


async def is_free_trial_active(user_address=None):
    trial_end_date = GLOBAL_TRIAL_START_DATE + timedelta(weeks=6)
    now = datetime.now()
    is_active = now < trial_end_date
    
    trial_info = {
        'is_active': is_active,
        'time_left': {
            'days': 0,
            'hours': 0,
            'minutes': 0,
            'seconds': 0
        },
        'image_count': TRIAL_IMAGE_COUNT if is_active else 0,
        'video_count': TRIAL_VIDEO_COUNT if is_active else 0
    }
    
    if is_active:
        time_left = trial_end_date - now
        days = time_left.days
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        trial_info['time_left'] = {
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds
        }

    if user_address and is_active:
        free_trial_override = await app.redis_client.get(f"user_{user_address}_free_trial_override")
        if free_trial_override == 'True':
            trial_info['is_active'] = False
            trial_info['image_count'] = 0
            trial_info['video_count'] = 0

    return trial_info['is_active'], trial_info['time_left'], trial_info['image_count'], trial_info['video_count']



async def get_or_initialize_user_data(
    user_prefix,
    free_trial_active,
    user_address,
    decrement_images=False,
    decrement_videos=False
):
    app.logger.info(f"Initializing/getting data for user {user_address}. Free trial active: {free_trial_active}")

    if is_whitelisted(user_address):
        app.logger.info(f"User {user_address} is whitelisted")
        return UNLIMITED_IMAGES, UNLIMITED_VIDEOS, 'Unlimited'

    # Retrieve user data from Redis
    user_initialized = await app.redis_client.get(f"{user_prefix}initialized")
    images_left = int(await app.redis_client.get(f"{user_prefix}images_left") or 0)
    videos_left = int(await app.redis_client.get(f"{user_prefix}videos_left") or 0)
    tier_status = await app.redis_client.get(f"{user_prefix}tier") or 'None'
    free_trial_override = await app.redis_client.get(f"{user_prefix}free_trial_override") or 'False'

    app.logger.info(
        f"Initial state for {user_address}: initialized={user_initialized}, "
        f"images_left={images_left}, videos_left={videos_left}, "
        f"tier_status={tier_status}, free_trial_override={free_trial_override}"
    )

    # Verify the user's actual tier based on their token holdings
    actual_tier_status, meets_requirements = await verify_tier_for_user(user_address)
    previous_tier_status = tier_status
    tier_status = actual_tier_status  # Update tier_status to the actual tier

    # Check if the tier has changed
    tier_changed = previous_tier_status != tier_status

    # Get the images and videos per month for previous and current tiers
    previous_images_total = SUBSCRIPTION_PLANS.get(previous_tier_status, {}).get('images_per_month', 0)
    previous_videos_total = SUBSCRIPTION_PLANS.get(previous_tier_status, {}).get('videos_per_month', 0)
    current_images_total = SUBSCRIPTION_PLANS.get(tier_status, {}).get('images_per_month', 0)
    current_videos_total = SUBSCRIPTION_PLANS.get(tier_status, {}).get('videos_per_month', 0)

    if free_trial_active and free_trial_override != 'True':
        # Handle free trial logic
        app.logger.info(f"Free trial is active for {user_address}")
        if not user_initialized or previous_tier_status != 'Free Trial':
            # Set trial counts during first initialization or if tier has changed
            _, _, trial_images, trial_videos = await is_free_trial_active(user_address)
            images_left = trial_images
            videos_left = trial_videos
            tier_status = 'Free Trial'
            app.logger.info(f"Initializing trial user with {images_left} images and {videos_left} videos")
            await app.redis_client.set(f"{user_prefix}initialized", 'True')
            await app.redis_client.set(f"{user_prefix}tier", tier_status)
            await app.redis_client.set(f"{user_prefix}images_left", str(images_left))
            await app.redis_client.set(f"{user_prefix}videos_left", str(videos_left))
        else:
            # Keep existing counts for trial users
            app.logger.info(f"Using existing trial counts for {user_address}: {images_left} images and {videos_left} videos")
    else:
        # Non-free trial user
        if not user_initialized:
            # First time initialization
            images_left = current_images_total
            videos_left = current_videos_total
            await app.redis_client.set(f"{user_prefix}initialized", 'True')
            await app.redis_client.set(f"{user_prefix}images_left", str(images_left))
            await app.redis_client.set(f"{user_prefix}videos_left", str(videos_left))
            await app.redis_client.set(f"{user_prefix}tier", tier_status)
            app.logger.info(f"Initialized new user {user_address} with tier {tier_status}, images_left={images_left}, videos_left={videos_left}")
        elif tier_changed:
            # Adjust counts based on new tier
            app.logger.info(f"Tier changed for user {user_address} from {previous_tier_status} to {tier_status}")
            # Calculate images and videos used so far
            images_used = previous_images_total - images_left
            videos_used = previous_videos_total - videos_left

            app.logger.info(f"User {user_address} has used {images_used} images and {videos_used} videos so far")

            # Calculate new images_left and videos_left
            images_left = max(current_images_total - images_used, 0)
            videos_left = max(current_videos_total - videos_used, 0)

            await app.redis_client.set(f"{user_prefix}images_left", str(images_left))
            await app.redis_client.set(f"{user_prefix}videos_left", str(videos_left))
            await app.redis_client.set(f"{user_prefix}tier", tier_status)
            app.logger.info(
                f"Adjusted counts for {user_address}: images_left={images_left}, videos_left={videos_left}"
            )
        else:
            app.logger.info(f"No tier change for {user_address}. Counts remain the same.")

    # Handle image decrement
    if decrement_images:
        if images_left > 0 or images_left == UNLIMITED_IMAGES:
            if images_left != UNLIMITED_IMAGES:
                images_left -= 1
                await app.redis_client.set(f"{user_prefix}images_left", str(images_left))
            app.logger.info(f"Decremented images for {user_address}. New images_left: {images_left}")
        else:
            app.logger.warning(f"User {user_address} has no images left to decrement.")
            raise ValueError("No images left to use")

    # Handle video decrement
    if decrement_videos:
        has_video_access = (
            tier_status in ['Optix Blend', 'Optix Pro', 'Optix Elite'] or
            free_trial_active
        )
        if has_video_access:
            if videos_left > 0 or videos_left == UNLIMITED_VIDEOS:
                if videos_left != UNLIMITED_VIDEOS:
                    videos_left -= 1
                    await app.redis_client.set(f"{user_prefix}videos_left", str(videos_left))
                app.logger.info(f"Decremented videos for {user_address}. New videos_left: {videos_left}")
            else:
                app.logger.warning(f"User {user_address} has no videos left to decrement.")
                raise ValueError("No video generations left")
        else:
            app.logger.warning(
                f"User {user_address} attempted to use video generation without proper access"
            )
            raise ValueError("Video generation requires Optix Blend tier or higher subscription")

    app.logger.info(
        f"Final state for {user_address}: images_left={images_left}, videos_left={videos_left}, tier_status={tier_status}"
    )
    return images_left, videos_left, tier_status

async def update_moralis_price_job():
    try:
        usd_price_per_token = get_token_price_from_moralis()  # calls Moralis
        contract = get_token_contract()
        total_supply = contract.functions.totalSupply().call()
        token_decimals = contract.functions.decimals().call()
        total_supply_adjusted = Decimal(total_supply) / Decimal(10 ** token_decimals)

        market_cap = total_supply_adjusted * usd_price_per_token

        await app.redis_client.set("moralis_usd_price", str(usd_price_per_token))
        await app.redis_client.set("moralis_market_cap", str(market_cap))
        await app.redis_client.set("moralis_last_updated", str(datetime.now().timestamp()))
        
        logging.info(f"[update_moralis_price_job] Updated price = {usd_price_per_token}, market cap = {market_cap}")
    except Exception as e:
        logging.error(f"[update_moralis_price_job] Error: {e}")


async def reset_monthly_counts():
    logging.info("Starting monthly image and video count reset")
    async with app.redis_client.pipeline(transaction=True) as pipe:
        # Get all user keys
        user_keys = await app.redis_client.keys("user_*_tier")
        
        for user_key in user_keys:
            user_prefix = user_key.rsplit('_', 1)[0] + '_'
            user_address = user_prefix.split('_')[1]
            current_tier = await app.redis_client.get(user_key)
            
            if is_whitelisted(user_address):
                pipe.set(f"{user_prefix}images_left", str(UNLIMITED_IMAGES))
                pipe.set(f"{user_prefix}videos_left", str(UNLIMITED_VIDEOS))
                pipe.set(f"{user_prefix}tier", 'Unlimited')
                logging.info(f"Reset limits for whitelisted user: {user_address}")
                continue

            # Check trial status first
            free_trial_active, _, trial_images, trial_videos = await is_free_trial_active(user_address)
            if free_trial_active:
                pipe.set(f"{user_prefix}images_left", str(trial_images))
                pipe.set(f"{user_prefix}videos_left", str(trial_videos))
                pipe.set(f"{user_prefix}tier", 'Free Trial')
                logging.info(f"Reset limits for trial user: {user_address}, images: {trial_images}, videos: {trial_videos}")
                continue

            # Verify if the user still meets the requirements for their current tier
            tier, meets_requirements = await verify_tier_for_user(user_address)

            if meets_requirements and tier == current_tier:
                # User still qualifies for their current tier
                new_image_count = SUBSCRIPTION_PLANS[current_tier]['images_per_month']
                new_video_count = SUBSCRIPTION_PLANS[current_tier]['videos_per_month']
                pipe.set(f"{user_prefix}images_left", str(new_image_count))
                pipe.set(f"{user_prefix}videos_left", str(new_video_count))
                logging.info(f"Reset limits for user: {user_address}, maintaining tier: {current_tier}, images: {new_image_count}, videos: {new_video_count}")
            elif meets_requirements and tier in SUBSCRIPTION_PLANS:
                # User's tier has changed (upgraded or downgraded)
                new_image_count = SUBSCRIPTION_PLANS[tier]['images_per_month']
                new_video_count = SUBSCRIPTION_PLANS[tier]['videos_per_month']
                pipe.set(f"{user_prefix}images_left", str(new_image_count))
                pipe.set(f"{user_prefix}videos_left", str(new_video_count))
                pipe.set(f"{user_prefix}tier", tier)
                logging.info(f"Updated tier for user: {user_address}, new tier: {tier}, images: {new_image_count}, videos: {new_video_count}")
            else:
                # User no longer meets requirements for any tier
                pipe.set(f"{user_prefix}images_left", '0')
                pipe.set(f"{user_prefix}videos_left", '0')
                pipe.set(f"{user_prefix}tier", 'None')
                logging.info(f"User {user_address} no longer meets tier requirements. Set counts to 0.")

            pipe.set(f"{user_prefix}last_reset", str(datetime.now().timestamp()))
        
        # Execute all commands in the pipeline
        await pipe.execute()
    
    logging.info("Monthly image and video count reset completed")



# Set up the scheduler
scheduler = AsyncIOScheduler()
scheduler.add_job(reset_monthly_counts, CronTrigger(day=1, hour=0, minute=0))
scheduler.add_job(update_moralis_price_job, CronTrigger(hour='*/6'))

async def create_redis_client():
    return Redis(
        host='redis-11850.c270.us-east-1-3.ec2.redns.redis-cloud.com',  
        port=11850,         
        password='ycwsjYErV556fuZ1aPGDog4lzBsAa5tp',              
        decode_responses=True  
    )

async def start_app():
    # Initialize any necessary components before running the server
    app.redis_client = await create_redis_client()
    scheduler.start()
    # Here you can add any other initialization you need

async def main():
    await start_app()
    hyper_config = Config()
    hyper_config.bind = ["0.0.0.0:" + os.getenv('PORT', '5000')]
    await serve(app, hyper_config)

if __name__ == "__main__":
    asyncio.run(main())
