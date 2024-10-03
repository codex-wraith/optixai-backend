from quart import Quart, jsonify, request, make_response, current_app
from quart_cors import cors
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
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
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
GLOBAL_TRIAL_START_DATE = datetime(2024, 9, 30)
UNLIMITED_IMAGES = -1 
WHITELISTED_ADDRESSES = [
    "0xe3dCD878B779C959A68fE982369E4c60c7503c38",  
    "0x780AfC062519614C83f1DbF9B320345772139e1e",
    "0xf52AfD0fF44aCfF80e9b3e54fe577E25af3f396E",
    "0xc6ca21DFfa38Fd88cd2bBd5D249CcD49F2346023",
    "0x722b7C259fafFe4fb060745EE5a5FdE5EAA7F00E"
]
SUBSCRIPTION_PLANS = {
    'Tier 1': {'percentage': Decimal('0.5'), 'images_per_month': 50},
    'Tier 2': {'percentage': Decimal('1.0'), 'images_per_month': 100},
    'Tier 3': {'percentage': Decimal('1.5'), 'images_per_month': 1000},
}
app = Quart(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = Redis.from_url(os.environ.get('REDISCLOUD_URL'))
cors(app,
     allow_origin=['https://www.pixl-ai.io', 'https://pixl-ai.io', 'https://pixl-ai-io.github.io/bundletest', 'https://pixl-ai-07c92c.webflow.io'],
     allow_methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'User-Address'],
     expose_headers=['Content-Type', 'User-Address'],
     allow_credentials=True)

@app.route('/tiers-info', methods=['GET', 'OPTIONS'])
async def tiers_info():
    if request.method == 'OPTIONS':
        return await handle_options_request()
    
    try:
        contract = get_token_contract()
        total_supply = contract.functions.totalSupply().call()
        token_decimals = contract.functions.decimals().call()
        total_supply_adjusted = Decimal(total_supply) / Decimal(10 ** token_decimals)
        
        tiers_info = {}
        for tier, plan in SUBSCRIPTION_PLANS.items():
            percentage = plan['percentage']  # Already a Decimal
            tokens_required = (percentage / Decimal(100)) * total_supply_adjusted
            tiers_info[tier] = {
                'percentage': float(percentage),  # Convert to float for JSON serialization
                'tokensRequired': float(tokens_required),
                'imagesPerMonth': plan['images_per_month']
            }
        
        return jsonify({
            'success': True,
            'totalSupply': float(total_supply_adjusted),
            'tiers': tiers_info
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

@app.route('/proxy-image')
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
            tier='Unlimited',
            availableUpgrades=[]
        )
    
    # Ensure Redis client is initialized
    if not app.redis_client:
        app.redis_client = await create_redis_client()

    user_prefix = f"user_{user_address}_"
    free_trial_active, remaining_time = await is_free_trial_active(user_address)

    if request.method == 'POST':
        images_left, tier_status = await get_or_initialize_user_data(user_prefix, free_trial_active, user_address)
        if images_left is None:
            return jsonify({'error': 'No images left to use'}), 400

        actual_tier_status, meets_requirements = await verify_tier_for_user(user_address)
        
        if actual_tier_status != 'None' and meets_requirements and free_trial_active:
            # User upgraded during free trial and meets requirements
            await app.redis_client.set(f"user_{user_address}_free_trial_override", 'False')
            free_trial_active = False
            tier_status = actual_tier_status
        elif free_trial_active:
            tier_status = 'Free Trial'
        else:
            tier_status = actual_tier_status

        await app.redis_client.set(f"{user_prefix}tier", tier_status)

    else:
        # For GET requests, fetch current data
        images_left = int(await app.redis_client.get(f"{user_prefix}images_left") or 0)
        tier_status = await app.redis_client.get(f"{user_prefix}tier") or 'None'
        if free_trial_active and tier_status == 'None':
            tier_status = 'Free Trial'

    # Determine available upgrades
    all_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    if tier_status == 'Free Trial':
        available_upgrades = all_tiers
    else:
        current_tiers = tier_status.split(', ')
        available_upgrades = [tier for tier in all_tiers if tier not in current_tiers]

    return jsonify(
        success=True,
        freeTrialActive=free_trial_active,
        trialTimeLeft=remaining_time,
        imagesLeft=images_left,
        tier=tier_status,
        availableUpgrades=available_upgrades
    )

@app.route('/trial-status', methods=['GET', 'OPTIONS'])
async def get_trial_status():
    if request.method == 'OPTIONS':
        return '', 204

    user_address = request.headers.get('User-Address')
    free_trial_active, remaining_time = await is_free_trial_active(user_address)

    # Return the free trial status as JSON
    return jsonify({
        'freeTrialActive': free_trial_active,
        'trialTimeLeft': remaining_time
    })

@app.route('/generate', methods=['POST', 'OPTIONS'])
async def generate_content():
    if request.method == 'OPTIONS':
        return await handle_options_request()

    try:
        data = await request.get_json()
        # Extract aspect ratio, default to '1:1' if not provided
        aspect_ratio = data.get('aspectRatio', '1:1')

        # Map aspect ratio values if necessary
        aspect_ratio_mapping = {
            '1:1': '1:1',
            '16:9': '16:9',
            '9:16': '9:16',
            '3:2': '3:2',
            '2:3': '2:3',
            '5:4': '5:4',
        }
        selected_aspect_ratio = aspect_ratio_mapping.get(aspect_ratio, '1:1')
    
        prompt_used = None
        if data.get('tier3Prompt'):
            prompt_used = data['tier3Prompt']
            tier_used = 'Tier 3'
        elif data.get('tier2Prompt'):
            prompt_used = data['tier2Prompt']
            tier_used = 'Tier 2'
        elif data.get('tier1Prompt'):
            prompt_used = data['tier1Prompt']
            tier_used = 'Tier 1'
        else:
            return jsonify({'error': 'No valid prompt provided'}), 400

        logging.info(f"Original prompt ({tier_used}): {prompt_used}")

        prompts = {
            'Tier 1': f"""Refine this prompt for a pixel art style image with cartoon elements: '{prompt_used}'
            Output a single, cohesive prompt that incorporates these elements:
            - Pixel art aesthetics (e.g., low resolution, limited color palette)
            - Cartoon-style elements
            - Key compositional elements (foreground, background)
            - Mood or atmosphere
            Do not include:
            - Technical instructions (e.g., resolution numbers, color codes)
            - Mentions of UI elements, borders, or non-artistic components
            - Realistic or photographic elements
            Format the output as a single, descriptive sentence without bullet points or separate instructions.""",

            'Tier 2': f"""Refine this prompt for an image blending pixel art (40%) with photorealistic (60%) elements: '{prompt_used}'
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

            'Tier 3': f"""Enhance this prompt for a photorealistic image with subtle artistic touches: '{prompt_used}'
            Craft a single, detailed prompt that includes:
            - Specific lighting details (direction, intensity, color)
            - Subtle artistic textures that maintain realism
            - Key compositional elements (foreground, background)
            - Mood and atmosphere
            Ensure to:
            - Keep the original intent and key elements of the prompt
            - Avoid mentioning non-photorealistic or obvious digital effects
            Output a single, comprehensive sentence that guides the AI to create a highly detailed, photorealistic image with artistic nuances."""
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

        # Use Replicate Flux Pro API for all tiers
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input={
                "prompt": final_prompt,
                "guidance": 3.5,
                "interval": 2,
                "aspect_ratio": selected_aspect_ratio,
                "output_format": "png",
                "output_quality": 100,
                "safety_tolerance": 5
            },
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
        user_address = data.get('userAddress')
        is_whitelisted_user = is_whitelisted(user_address)
        if not is_whitelisted_user:
           user_prefix = f"user_{user_address}_"
           free_trial_active, _ = await is_free_trial_active(user_address)
    
           # First, get the current image count without decrementing
           images_left, _ = await get_or_initialize_user_data(user_prefix, free_trial_active, user_address, decrement=False)
    
           # Check if there are images left
           if images_left <= 0:
              return jsonify({'error': 'No images left to use'}), 400
    
           # If we have images, then decrement and get the new count
           images_left, _ = await get_or_initialize_user_data(user_prefix, free_trial_active, user_address, decrement=True)
        else:
         images_left = UNLIMITED_IMAGES
        
        return jsonify({
            'text': final_prompt,
            'file_id': file_id, 
            'imagesLeft': images_left,  # Reflect updated image count or UNLIMITED_IMAGES for whitelisted users
        })

    except Exception as e:
        app.logger.error(f"Error generating content: {str(e)}")
        return jsonify({'error': str(e)}), 500


async def verify_tier_for_user(user_address):
    if is_whitelisted(user_address):
        return 'Unlimited', True

    try:
        checksum_address = Web3.to_checksum_address(user_address)
        percentage_held = verify_user_holdings(checksum_address)
        
        for tier, plan in SUBSCRIPTION_PLANS.items():
            if percentage_held >= plan['percentage']:
                return tier, True
        
        return 'None', False
    except Exception as e:
        logging.error(f"Error verifying tier for user {user_address}: {str(e)}")
        return 'None', False


def get_web3_instance():
    providers = [
        Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{INFURA_API_KEY}'),
        # You can add backup providers here, e.g.:
        # Web3.HTTPProvider('https://eth-mainnet.alchemyapi.io/v2/YOUR-ALCHEMY-KEY'),
    ]

    for provider in providers:
        try:
            web3 = Web3(provider)
            if web3.is_connected():
                return web3
        except Web3Exception as e:
            print(f"Failed to connect to {provider.endpoint_uri}: {e}")
    
    raise ConnectionError("Failed to connect to any Ethereum provider")

def get_token_contract():
    token_address = "0x6982508145454ce325ddbe47a25d4ec3d2311933"
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

async def generate_text(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    # Log the refined prompt
    logging.info(f"Refined prompt: {response.text}")
    
    return response

async def handle_options_request():
    """Handle OPTIONS request for CORS."""
    response = await make_response('', 204)
    response.headers['Access-Control-Allow-Origin'] = 'https://www.pixl-ai.io, https://pixl-ai.io, https://pixl-ai-io.github.io/bundletest, https://pixl-ai-07c92c.webflow.io'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, User-Address'
    return response

async def is_free_trial_active(user_address=None):
    trial_end_date = GLOBAL_TRIAL_START_DATE + timedelta(weeks=1)
    now = datetime.now()
    is_active = now < trial_end_date
    
    if is_active:
        time_left = trial_end_date - now
        days = time_left.days
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        remaining_time = {
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds
        }
    else:
        remaining_time = {
            'days': 0,
            'hours': 0,
            'minutes': 0,
            'seconds': 0
        }

    if user_address and is_active:
        free_trial_override = await app.redis_client.get(f"user_{user_address}_free_trial_override")
        if free_trial_override == True:
            is_active = False
            remaining_time = {
                'days': 0,
                'hours': 0,
                'minutes': 0,
                'seconds': 0
            }

    return is_active, remaining_time


async def get_or_initialize_user_data(user_prefix, free_trial_active, user_address, decrement=False):
    app.logger.info(f"Initializing/getting data for user {user_address}. Free trial active: {free_trial_active}")

    if is_whitelisted(user_address):
        app.logger.info(f"User {user_address} is whitelisted")
        return UNLIMITED_IMAGES, 'Unlimited'

    user_initialized = await app.redis_client.get(f"{user_prefix}initialized")
    images_left_str = await app.redis_client.get(f"{user_prefix}images_left") or '0'
    images_left = int(images_left_str)
    tier_status = await app.redis_client.get(f"{user_prefix}tier") or 'None'
    free_trial_override = await app.redis_client.get(f"{user_prefix}free_trial_override") or 'False'

    app.logger.info(f"Initial state for {user_address}: initialized={user_initialized}, images_left={images_left}, tier_status={tier_status}, free_trial_override={free_trial_override}")

    if free_trial_active and free_trial_override != 'True':
        app.logger.info(f"Free trial is active for {user_address}")
        if user_initialized is None:
            images_left = 20  # Initialize with 20 images only if not initialized
            tier_status = 'Free Trial'
    elif user_initialized is None:
        app.logger.info(f"Initializing non-free trial user {user_address}")
        actual_tier_status, meets_requirements = await verify_tier_for_user(user_address)
        if meets_requirements:
            tier_status = actual_tier_status
            images_left = SUBSCRIPTION_PLANS.get(tier_status, {}).get('images_per_month', 0)
        else:
            tier_status = 'None'
            images_left = 0
        app.logger.info(f"Non-free trial user {user_address} initialized with tier {tier_status} and {images_left} images")

    if user_initialized is None:
        await app.redis_client.set(f"{user_prefix}initialized", 'True')
        await app.redis_client.set(f"{user_prefix}tier", tier_status)
        await app.redis_client.set(f"{user_prefix}images_left", str(images_left))
        app.logger.info(f"User {user_address} marked as initialized with {images_left} images and tier {tier_status}")

    if decrement and images_left > 0:
        images_left -= 1
        await app.redis_client.set(f"{user_prefix}images_left", str(images_left))
        app.logger.info(f"Decremented images for {user_address}. New count: {images_left}")

    app.logger.info(f"Final state for {user_address}: images_left={images_left}, tier_status={tier_status}")
    return images_left, tier_status


async def reset_monthly_image_count():
    logging.info("Starting monthly image count reset")
    async with app.redis_client.pipeline(transaction=True) as pipe:
        # Get all user keys
        user_keys = await app.redis_client.keys("user_*_tier")
        
        for user_key in user_keys:
            user_prefix = user_key.rsplit('_', 1)[0] + '_'
            user_address = user_prefix.split('_')[1]
            current_tier = await app.redis_client.get(user_key)
            
            if is_whitelisted(user_address):
                pipe.set(f"{user_prefix}images_left", str(UNLIMITED_IMAGES))
                pipe.set(f"{user_prefix}tier", 'Unlimited')
                logging.info(f"Reset image limit for whitelisted user: {user_address}")
                continue

            if current_tier in SUBSCRIPTION_PLANS:
                # Verify if the user still meets the requirements for their current tier
                tier, meets_requirements = await verify_tier_for_user(user_address)
                
                if meets_requirements and tier == current_tier:
                    # User still qualifies for their current tier
                    new_image_count = SUBSCRIPTION_PLANS[current_tier]['images_per_month']
                    pipe.set(f"{user_prefix}images_left", str(new_image_count))
                    logging.info(f"Reset image limit for user: {user_address}, maintaining tier: {current_tier}, images: {new_image_count}")
                else:
                    # User's tier has changed, perform full verification
                    if meets_requirements and tier in SUBSCRIPTION_PLANS:
                        new_image_count = SUBSCRIPTION_PLANS[tier]['images_per_month']
                        pipe.set(f"{user_prefix}images_left", str(new_image_count))
                        pipe.set(f"{user_prefix}tier", tier)
                        logging.info(f"Updated tier for user: {user_address}, new tier: {tier}, images: {new_image_count}")
                    else:
                        # User no longer meets requirements for any tier
                        pipe.set(f"{user_prefix}images_left", '0')
                        pipe.set(f"{user_prefix}tier", 'None')
                        logging.info(f"User {user_address} no longer meets tier requirements. Set images to 0.")
            else:
                # Handle users not in a subscription plan (e.g., free trial or no active plan)
                free_trial_active, _ = await is_free_trial_active(user_address)
                if free_trial_active:
                    free_trial_images = 20  # Or whatever your free trial limit is
                    pipe.set(f"{user_prefix}images_left", str(free_trial_images))
                    pipe.set(f"{user_prefix}tier", 'Free Trial')
                    logging.info(f"Reset image limit for active free trial user: {user_address}, images: {free_trial_images}")
                else:
                    # Check if user now qualifies for a tier
                    tier, meets_requirements = await verify_tier_for_user(user_address)
                    if meets_requirements and tier in SUBSCRIPTION_PLANS:
                        new_image_count = SUBSCRIPTION_PLANS[tier]['images_per_month']
                        pipe.set(f"{user_prefix}images_left", str(new_image_count))
                        pipe.set(f"{user_prefix}tier", tier)
                        logging.info(f"User {user_address} now qualifies for tier: {tier}, images: {new_image_count}")
                    else:
                        pipe.set(f"{user_prefix}images_left", '0')
                        pipe.set(f"{user_prefix}tier", 'None')
                        logging.info(f"No active tier or free trial for user: {user_address}. Set images to 0.")
            
            pipe.set(f"{user_prefix}last_reset", str(datetime.now().timestamp()))
        
        # Execute all commands in the pipeline
        await pipe.execute()
    
    logging.info("Monthly image count reset completed")

# Set up the scheduler
scheduler = AsyncIOScheduler()
scheduler.add_job(reset_monthly_image_count, CronTrigger(day=1, hour=0, minute=0))


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
