import json
import re
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def clean_text(text):
    """Removes unsupported surrogate Unicode characters."""
    return text.encode("utf-8", "ignore").decode("utf-8")


def extract_json(text):
    """Extracts JSON from LLM response, removing extra text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))  # Ensure valid JSON parsing
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {e}")
            return {}
    return {}



def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should follow the title case convention. Example: "Motivation", "Job Search"
    3. **Only output a valid JSON object**. Do not include any extra text or explanation.
    4. Output should have a mapping of the original tag and the unified tag. 

    Here is the list of tags: 
    {tags}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})

    try:
        json_parser = JsonOutputParser()
        extracted_json = extract_json(response.content)  # Extract JSON safely
        return extracted_json
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse jobs.")


def extract_metadata(post):
    """Extracts metadata (line count, language, tags) from a LinkedIn post using an LLM."""
    post = clean_text(post)  # Clean the post text before sending to LLM

    template = '''
        You are given a LinkedIn post. Extract number of lines, language, and tags.
        1. Respond with **ONLY** a valid JSON object. No extra text, no preamble.
        2. JSON keys: line_count (int), language (English/Hinglish), tags (array of max 2).

        Post: {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm

    try:
        response = chain.invoke(input={"post": post})
        res = extract_json(response.content.strip())

        if isinstance(res, dict):
            return res
        else:
            print(f"Warning: Unexpected metadata format: {res}")
            return {"line_count": 0, "language": "Unknown", "tags": []}

    except (OutputParserException, json.JSONDecodeError) as e:
        print(f"Output parsing failed: {e}")
        return {"line_count": 0, "language": "Unknown", "tags": []}


def process_posts(raw_file_path, processed_file_path="data/processed_posts.json"):
    """Processes raw LinkedIn posts and saves enriched posts with metadata."""
    enriched_posts = []

    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        for post in posts:
            post['text'] = clean_text(post.get('text', ''))  # Clean text before processing
            metadata = extract_metadata(post['text'])

            if isinstance(metadata, dict):
                post_with_metadata = {**post, **metadata}  # Merge dicts safely
            else:
                print(f"Warning: metadata is not a dictionary: {metadata}")
                post_with_metadata = post  # Skip merging if metadata is invalid

            # Ensure every post has a "tags" key to avoid KeyError
            post_with_metadata['tags'] = post_with_metadata.get('tags', [])

            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags.get(tag, tag) for tag in current_tags}  # Keep tag if not found in mapping
        post['tags'] = list(new_tags)

    # Save JSON with `ensure_ascii=False` to support non-ASCII characters safely
    with open(processed_file_path, "w", encoding="utf-8") as outfile:
        json.dump(enriched_posts, outfile, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    process_posts("C:/LinkedIn/data/raw_posts.json", "C:/LinkedIn/data/processed_posts.json")
