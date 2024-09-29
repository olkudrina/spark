import sys
from awsglue.transforms import *
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import boto3
import logging
import time
import requests
import re
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


def fetch_memes(subreddit='memes', count=20):
    meme_list = []
    api_url = f"https://meme-api.com/gimme/{subreddit}/{count}"

    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            memes = data.get('memes', [])
            meme_list.extend(memes)
        else:
            print(f"couldn't get results - {response.status_code}")
    except Exception as e:
        print(e)
    
    return meme_list

def download_image(url):
    try:
        response = requests.get(url)
        return BytesIO(response.content)
    except:
        return None

def generate_random_one_word(image_bytes, processor, model):
    try:
        image = Image.open(image_bytes)
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # get a random word
        all_words = [re.sub('\W+','', i) for i in caption.split()]
        pos = np.random.randint(0, len(all_words))

        return caption.split()[pos] if caption else "impossibletodescribe"
    except Exception as e:
        return 'notprocessed'

def generate_one_word(image_bytes, processor, model):
    try:
        image = Image.open(image_bytes)
        condition_text = 'one word for the image is '
        inputs = processor(images=image, text=condition_text, return_tensors="pt")
        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True)
        result = result.replace(condition_text, '').strip()
        return result if result else "impossible_to_describe"
    except Exception as e:
        return 'notprocessed'

def generate_name(meme_objects):
    description, one_word, obj_ = meme_objects
    
    file_name = re.sub('\W+', '_', description).lower() + ".png"
    return (one_word, file_name, obj_)

def save_to_s3(partitioned_data, bucket_name):
    partition, file_name, image_bytes = partitioned_data
    s3_client = boto3.client('s3')

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"{partition}/{file_name}",
            Body=image_bytes.getvalue()
        )
        logger.info(f"uploaded {file_name} to {partition}")
    except Exception as e:
        logger.info(f"failed {file_name}: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    sc = SparkContext.getOrCreate()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    all_memes = []
    for i in range(5):
        temp = fetch_memes()
        all_memes.extend([(i['title'], i['url']) for i in temp])
        time.sleep(0.1)
    
    rdd_urls = sc.parallelize(all_memes)
    logger.info(f'Total number of urls: {rdd_urls.count()}')
    logger.info(f'Example: {rdd_urls.take(1)}')
    
    rdd_images = (
        rdd_urls
        .map(lambda x: (x[0], download_image(x[1])))
        .filter(lambda x: x[1] is not None)
    )
    logger.info(f'Total number of image objects: {rdd_images.count()}')
    logger.info(f'Example: {rdd_images.take(1)}')

    processed = rdd_images.map(
        lambda x: (x[0], generate_random_one_word(x[1], processor, model), x[1])
    )
    processed = processed.map(generate_name).filter(lambda x: x is not None)
    logger.info(f'Processed example: {processed.take(1)}')

    processed.persist()
    logger.info('start writing to s3')
    processed.foreach(lambda x: save_to_s3(x, 'my-memes'))

    job.commit()
