import re
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langsmith import traceable
from tqdm import tqdm
from typing import List, Dict, Any

from prompts import *


def read_json_file(file_path: str) -> List[Dict[Any, Any]]:
    """Reads and returns data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(data: List[Dict[Any, Any]], file_path: str) -> None:
    """Saves data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def preprocess_data(input_path: str, output_path: str, min_post_length: int = 10) -> str:
    """
    Preprocess data from input JSON file based on source type and minimum post length.

    Args:
        input_path (str): Path to input JSON file
        output_path (str): Path where to save processed data
        source (str): Source type (e.g., "tg" for Telegram)
        min_post_length (int): Minimum length for posts to be included

    Returns:
        str: Path to the processed output file
    """
    input_json = read_json_file(input_path)
    posts_list = []

    for post in input_json:
        # Escape backslashes and filter by length
        post["text"] = post["text"].replace('\\', '\\\\')
        if post["text"] and len(post["text"]) > min_post_length:
            # Remove unnecessary fields
            for field in ["forwarded_from", "reactions", "views"]:
                post.pop(field, None)
            posts_list.append(post)

    save_json_file(posts_list, output_path)
    return output_path


@traceable(name="Translate data", metadata={"model": "GPT-4o-mini"}, project_name="nasa-hackathon")
def translate_data(model, input_path: str, output_path: str, batch_size: int, asset) -> str:
    """
    Translate data using specified model, processing in batches.

    Args:
        model: Language model for translation
        input_path (str): Path to input JSON file
        output_path (str): Path where to save translated data
        source (str): Source type (e.g., "tg" for Telegram)
        batch_size (int): Number of items to process in each batch
        asset (Dict): Asset information for context

    Returns:
        str: Path to the translated output file
    """
    # Read input JSON file
    input_json = read_json_file(input_path)

    # Process data if the source is Telegram ("tg")
    output_json = input_json.copy()
    translate_chain = TRANSLATE_ENG | model | JsonOutputParser()

    # Translate channel name
    channel_name_ua = input_json[0]["channel_name"]
    channel_name_eng = translate_chain.invoke({"messages": [{"message_id": 1, "text": channel_name_ua}], "kpi_asset": asset})["messages"][0]["text"]
    print(channel_name_eng)

    # Create message_id to index mapping for efficient updates
    message_id_map = {msg.get("message_id"): i for i, msg in enumerate(output_json)}

    # Process messages in batches with progress bar
    for i in tqdm(range(0, len(input_json), batch_size)):
        batch = input_json[i:i + batch_size]
        translated_batch = translate_chain.invoke({"messages": batch, "kpi_asset": asset})

        # Update translations in output JSON
        for translated_post in translated_batch["messages"]:
            if (message_id := translated_post["message_id"]) in message_id_map:
                index = message_id_map[message_id]
                output_json[index].update({
                    "channel_name": channel_name_eng,
                    "text": translated_post["text"]
                })

    # Write translated data to output JSON file
    save_json_file(output_json, output_path)

    return output_path


@traceable(name="Extract news", metadata={"model": "GPT-4o"}, project_name="nasa-hackathon")
def extract_news(model, input_path: str, output_path: str, batch_size: int, save_neutral: bool,
                 asset) -> str:
    """
    Extract and classify news as positive, negative, or neutral.
    
    Args:
        model: The language model to use for classification
        input_path (str): Path to input JSON file
        output_path (str): Path to save processed results
        source (str): Source identifier
        batch_size (int): Number of items to process in each batch
        save_neutral (bool): Whether to include neutral classifications in output
        asset: Asset information for classification
        
    Returns:
        str: Path to the output file
    """
    # Load input data
    input_json = read_json_file(input_path)
    output_dict = []

    # Set up classification chain
    chain = EXTRACT_NEEDED | model | JsonOutputParser()

    # Process in batches with progress bar
    for i in tqdm(range(0, len(input_json), batch_size), total=len(input_json) // batch_size):
        batch = input_json[i:i + batch_size]
        classified_indices = chain.invoke({"messages": batch, "kpi_asset": asset})

        # Process classification results
        for item in input_json:
            if item["message_id"] in classified_indices["positive"]:
                item["sentiment"] = "positive"
                output_dict.append(item)
            elif item["message_id"] in classified_indices["negative"]:
                item["sentiment"] = "negative"
                output_dict.append(item)
            elif save_neutral:
                item["sentiment"] = "neutral"
                output_dict.append(item)

    # Save results
    save_json_file(output_dict, output_path)
    return output_path


@traceable(name="Classify news", metadata={"model": "GPT-4o"}, project_name="nasa-hackathon")
def classify_news(model, input_path: str, output_path: str, batch_size: int, asset) -> str:
    """
    Classify news items by topic based on their sentiment.
    
    Args:
        model: The language model to use for classification
        input_path (str): Path to input JSON file
        output_path (str): Path to save processed results
        source (str): Source identifier
        batch_size (int): Number of items to process in each batch
        asset: Asset information for classification
        
    Returns:
        str: Path to the output file
    """
    # Load and split input data by sentiment
    input_json = read_json_file(input_path)
    input_json_positive = [item for item in input_json if item.get("sentiment") == "positive"]
    input_json_negative = [item for item in input_json if item.get("sentiment") == "negative"]

    # Set up classification chains
    chain_positive = CLASSIFY_MESSAGES_POSITIVE | model | JsonOutputParser()
    chain_negative = CLASSIFY_MESSAGES_NEGATIVE | model | JsonOutputParser()

    # Process positive news
    for i in tqdm(range(0, len(input_json_positive), batch_size), 
                 total=len(input_json_positive) // batch_size):
        batch = input_json_positive[i:i + batch_size]
        classified_news_batch = chain_positive.invoke({"messages": batch, "kpi_asset": asset})
        
        # Update topics in original data
        for classified_item in classified_news_batch["news"]:
            for item in input_json:
                if item["message_id"] == classified_item["id"]:
                    item["topic"] = classified_item["topic"]

    # Process negative news
    for i in tqdm(range(0, len(input_json_negative), batch_size), 
                 total=len(input_json_negative) // batch_size):
        batch = input_json_negative[i:i + batch_size]
        classified_news_batch = chain_negative.invoke({"messages": batch, "kpi_asset": asset})
        
        # Update topics in original data
        for classified_item in classified_news_batch["news"]:
            for item in input_json:
                if item["message_id"] == classified_item["id"]:
                    item["topic"] = classified_item["topic"]

    # Save results
    save_json_file(input_json, output_path)
    return output_path


@traceable(name="Rephrase news", metadata={"model": "GPT-4o-mini"}, project_name="nasa-hackathon")
def rephrase_news(model, input_path: str, output_path: str, batch_size: int, asset) -> str:
    """
    Rephrase news items to create alternative versions.
    
    Args:
        model: The language model to use for rephrasing
        input_path (str): Path to input JSON file
        output_path (str): Path to save processed results
        source (str): Source identifier
        batch_size (int): Number of items to process in each batch
        asset: Asset information for processing
        
    Returns:
        str: Path to the output file
    """
    input_json = read_json_file(input_path)
    chain = REPHRASE_NEWS | model | JsonOutputParser()

    # Process in batches
    for i in tqdm(range(0, len(input_json), batch_size), total=len(input_json) // batch_size):
        batch = input_json[i:i + batch_size]
        rephrased_batch = chain.invoke({"messages": batch, "kpi_asset": asset})
        
        # Update rephrased content
        for rephrased_item in rephrased_batch["news"]:
            for item in input_json:
                if item["message_id"] == rephrased_item["id"]:
                    item["rephrased_news"] = rephrased_item["rephrased_news"]

    save_json_file(input_json, output_path)
    return output_path


@traceable(name="Add title", metadata={"model": "GPT-4o-mini"}, project_name="nasa-hackathon")
def add_titles(model, input_path: str, output_path: str, batch_size: int, asset) -> str:
    """
    Generate and add titles to news items.
    
    Args:
        model: The language model to use for title generation
        input_path (str): Path to input JSON file
        output_path (str): Path to save processed results
        source (str): Source identifier
        batch_size (int): Number of items to process in each batch
        asset: Asset information for processing
        
    Returns:
        str: Path to the output file
    """
    input_json = read_json_file(input_path)
    chain = GENERATE_TITLE | model | JsonOutputParser()

    # Process in batches
    for i in tqdm(range(0, len(input_json), batch_size), total=len(input_json) // batch_size):
        batch = input_json[i:i + batch_size]
        titled_batch = chain.invoke({"messages": batch, "kpi_asset": asset})
        
        # Add generated titles
        for titled_item in titled_batch["news"]:
            for item in input_json:
                if item["message_id"] == titled_item["id"]:
                    item["title"] = titled_item["title"]

    save_json_file(input_json, output_path)
    return output_path


@traceable(name="Extract location", metadata={"model": "GPT-4o"}, project_name="nasa-hackathon")
def extract_location(model, input_path: str, output_path: str, batch_size: int, asset) -> str:
    """
    Extract location information from news items.
    
    Args:
        model: The language model to use for location extraction
        input_path (str): Path to input JSON file
        output_path (str): Path to save processed results
        source (str): Source identifier
        batch_size (int): Number of items to process in each batch
        asset: Asset information for processing
        
    Returns:
        str: Path to the output file
    """
    input_json = read_json_file(input_path)
    chain = EXTRACT_LOCATION | model | JsonOutputParser()

    # Process in batches
    for i in tqdm(range(len(input_json) // batch_size), total=len(input_json) // batch_size):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = input_json[start_idx:end_idx]
        
        locations_batch = chain.invoke({"messages": batch, "kpi_asset": asset})
        
        # Add location information
        for post, location_info in zip(batch, locations_batch["news"]):
            post["location_type"] = location_info["location_type"]
            post["location"] = location_info["location"]

    save_json_file(input_json, output_path)
    return output_path


def add_location_fields(input_path: str, output_path: str) -> str:
    # Read input JSON file
    input_json = read_json_file(input_path)

    # Extract dormitory number
    channel_name = input_json[0]["channel_name"]
    numbers = re.findall(r'\d+', channel_name)
    dorm_number = int(numbers[0])

    kpi_dorms = {
        4: ['50.4490125', '30.4508679'],
        7: ['50.4485221', '30.4494086'],
        8: ['50.4364983', '30.4283648'],
        14: ['50.4496249', '30.4119686'],
        16: ['50.4471963', '30.4470161'],
        18: ['50.445927', '30.4450722'],
        19: ['50.4458979', '30.3652466'],
        20: ['50.4458397', '30.3652465']
    }

    # Add fields
    for post in input_json:
        post["location_type"] = "exact location"
        post["location"] = [f"KPI Dormitory {dorm_number}"]
        post['latitude'] = kpi_dorms[dorm_number][0]
        post['longitude'] = kpi_dorms[dorm_number][1]

    # Write updated data with location information to output JSON file
    save_json_file(input_json, output_path)

    return output_path


@traceable(name="Full pipeline", metadata={"version": "v.1.0.0", "model": "GPT-4o-mini"}, project_name="nasa-hackathon")
def main(files_list: list, gpt_4o_mini, gpt_4o, gpt_4o_mini_t02):
    """
    Main pipeline function that processes multiple files through various stages.
    
    Args:
        files_list: List of files to process
        gpt_4o_mini: GPT-4 mini model instance
        gpt_4o: GPT-4 model instance
        gpt_4o_mini_t02: GPT-4 mini model instance with different temperature
        
    The pipeline includes:
        1. Data preprocessing
        2. Translation
        3. News extraction
        4. News rephrasing
        5. Title generation
        6. News classification
        7. Location extraction/addition
    """
    for file in files_list:
        file_path = os.path.join(dir_path, file)
        
        # Determine if file is KPI-related
        search_terms = ["kpi", "dorm", "кпі", "гуртожиток", "кпи", "общежитие"]
        kpi_asset = {
            "name": "KPI (National Technical University of Ukraine 'Igor Sikorsky Kyiv Polytechnic Institute')",
            "description": "KPI, officially known as the National Technical University of Ukraine 'Igor Sikorsky Kyiv Polytechnic Institute', is one of Ukraine's largest technical universities. It is located in Kyiv, the capital of Ukraine."
        } if any(term in file.lower() for term in search_terms) else False

        # Step 1: Preprocess the data
        preprocessed_data_path = preprocess_data(input_path=file_path, output_path=f"{file_path[:-5]}_1_preprocessed.json", min_post_length=20)
        print(f"Step 1 done!")

        # Step 2: Translate the preprocessed data
        translated_data_path = translate_data(model=gpt_4o_mini, input_path=preprocessed_data_path, output_path=f"{file_path[:-5]}_2_translated.json", batch_size=40, asset=kpi_asset)
        print(f"Step 2 done!")

        # Step 3: Extract important news from the translated data
        important_data_path = extract_news(model=gpt_4o, input_path=translated_data_path, output_path=f"{file_path[:-5]}_3_pos_neg.json", batch_size=30, save_neutral=False, asset=kpi_asset)
        print(f"Step 3 done!")

        # Step 4: Rephrase the news
        rephrased_data_path = rephrase_news(model=gpt_4o_mini_t02, input_path=important_data_path, output_path=f"{file_path[:-5]}_4_rephrased.json", batch_size=30, asset=kpi_asset)
        print(f"Step 4 done!")

        # Step 5: Rephrase the news
        titled_data_path = add_titles(model=gpt_4o_mini_t02, input_path=rephrased_data_path, output_path=f"{file_path[:-5]}_5_titled.json", batch_size=30, asset=kpi_asset)
        print(f"Step 5 done!")

        # Step 6: Classify the news
        classified_data_path = classify_news(model=gpt_4o, input_path=titled_data_path, output_path=f"{file_path[:-5]}_6_classified.json", batch_size=30, asset=kpi_asset)
        print(f"Step 6 done!")

        if kpi_asset:
            # Step 7: Add location information
            locations_data = add_location_fields(input_path=classified_data_path, output_path=f"{file_path[:-5]}_7_locations.json")
            print(f"Step 7 done!")
        else:
            # Step 7: Extract location information
            locations_data = extract_location(model=gpt_4o, input_path=classified_data_path, output_path=f"{file_path[:-5]}_7_locations.json", batch_size=10, asset=kpi_asset)
            print(f"Step 7 done!")

if __name__ == "__main__":
    # Initialize environment and configurations
    load_dotenv()

    # Initialize GPT models with specific parameters
    gpt_4o_mini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0000000000000001,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    gpt_4o_mini_t02 = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.25,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    gpt_4o = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0000000000000001,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Set directory path and get file list
    dir_path = "" # put path here
    files_list = os.listdir(dir_path)

    # Execute main pipeline
    main(files_list, gpt_4o=gpt_4o, gpt_4o_mini=gpt_4o_mini, gpt_4o_mini_t02=gpt_4o_mini_t02)
