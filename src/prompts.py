from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ---------------------- TRANSLATE INTO ENGLISH ----------------------

TRANSLATE_ENG = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an expert translator with a strong focus on Ukrainian to English translations. Your primary responsibility is to deliver highly accurate, fluent English translations that maintain the original meaning, tone, context, and cultural nuances of the Ukrainian source text.

### Input Format
You will be given text fragments structured as follows:
[message_id]. [text fragment in ukrainian]
[message_id]. [text fragment in ukrainian]

### Output Structure
Return the translations in a JSON object, following this structure:
```json
{
    "messages": [  // A list of translated text fragments
        {
            "message_id": int,  // The unique identifier from the input
            "text": string  // The translated text fragment in English
        }
    ]
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Translate the following text fragments into English:
{% for message in messages %}
{{ message.message_id }}. {{ message.text }}
{% endfor %}
""", template_format='jinja2')
])

# ---------------------- EXTRACT POSITIVE / NEGATIVE ----------------------

EXTRACT_NEEDED = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an expert AI Assistant tasked with analyzing recent news about Kyiv city {% if kpi_asset %}or {{ kpi_asset.name }}{% endif %}.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.
{% if kpi_asset %}{{ kpi_asset.description }}{% endif %}

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Your task
Your goal is to analyze the sentiment of each news item provided. Based on the content, extract positive and negative news. The criteria for classifications are as follows:
1. **Negative**
Classify the news as Negative if it contains or emphasizes any of the following:
- **City Problems**: Infrastructure breakdowns, transportation disruptions, environmental hazards (e.g., pollution, resource depletion).
- **Citizen Discomfort**: Inconveniences such as strikes, road closures, or public service disruptions.
- **Accidents or Disasters**: Incidents like fires, floods, or other natural or man-made disasters.
- **Crime or Safety Concerns**: Reports of criminal activities, assaults, theft, or safety risks.
- **Social/Political Instability**: Protests, riots, political turmoil, or social unrest.
- **Economic Decline**: Rising unemployment, inflation, or other negative economic indicators.
- **Health Crises**: Outbreaks of diseases, healthcare system failures, or public health concerns.
- **Public Safety Risks**: Issues like unsafe neighborhoods or risks to public safety (e.g., building collapses).
- **Social Issues**: Reports on inequality, discrimination, housing shortages, or increases in the cost of living.
- **Deterioration in Infrastructure**: Decline in public services or neglect of city infrastructure.
- **Environmental concerns**: Pollution, climate change-related problems, resource depletion.

2. **Positive**
Classify the news as Positive if it contains or emphasizes any of the following:
- **Improvements in Infrastructure**: Development of new transportation systems, modern technologies, or other enhancements in public services.
- **Community Events or Achievements**: Celebrations, festivals, or charity efforts that boost morale and community spirit.
- **Economic Growth**: Reports of job creation, foreign investments, or reductions in unemployment.
- **Environmental Improvements**: Successful sustainability efforts or cleaner living conditions (e.g., better air quality).
- **Political Stability/Positive Reforms**: Effective policy changes, civic engagement, or signs of political stability.
- **Health and Wellness**: New healthcare initiatives, improved public health outcomes, or successful disease prevention efforts.
- **Social Progress**: Initiatives promoting equality, inclusivity, or a reduction in crime.
- **Cultural/Sporting Milestones**: Achievements in sports, culture, or science that foster pride and optimism.

#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
    "positive": [], // List of indexes of positively classified news
    "negative": []  // List of indexes of negative classified news
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Classify the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.text }}</news>
{% endfor %}
""", template_format='jinja2')
])

# ---------------------- CLASSIFY ----------------------

CLASSIFY_MESSAGES_NEGATIVE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an expert AI Assistant tasked with classifying recent news about Kyiv city {% if kpi_asset %}or {{ kpi_asset.name }}{% endif %}into specific topics. You will be provided with news articles or headlines, and your job is to determine the primary topic of each news item.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.
{% if kpi_asset %}{{ kpi_asset.description }}{% endif %}

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Your task
Your goal is to analyze each news item provided and classify it into one of the following topics:
- **City Problems**: Infrastructure breakdowns, transportation disruptions, environmental hazards (e.g., pollution, resource depletion).
- **Citizen Discomfort**: Inconveniences such as strikes, road closures, or public service disruptions.
- **Accidents or Disasters**: Incidents like fires, floods, or other natural or man-made disasters.
- **Crime or Safety Concerns**: Reports of criminal activities, assaults, theft, or safety risks.
- **Social/Political Instability**: Protests, riots, political turmoil, or social unrest.
- **Economic Decline**: Rising unemployment, inflation, or other negative economic indicators.
- **Health Crises**: Outbreaks of diseases, healthcare system failures, or public health concerns.
- **Public Safety Risks**: Issues like unsafe neighborhoods or risks to public safety (e.g., building collapses).
- **Social Issues**: Reports on inequality, discrimination, housing shortages, or increases in the cost of living.
- **Deterioration in Infrastructure**: Decline in public services or neglect of city infrastructure.
- **Environmental concerns**: Pollution, climate change-related problems, resource depletion.
- **Other**: News items that don't clearly fit into any of the above categories or contain multiple topics with no clear primary focus.

#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
  "news": [
    {
      "id": int,  // Unique identifier for the news item
      "topic": str,  // Primary topic of the news item
    }
  ]
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Classify the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.rephrased_news }}</news>
{% endfor %}
""", template_format='jinja2')
])

CLASSIFY_MESSAGES_POSITIVE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an expert AI Assistant tasked with classifying recent news about Kyiv city {% if kpi_asset %}or {{ kpi_asset.name }}{% endif %}into specific topics. You will be provided with news articles or headlines, and your job is to determine the primary topic of each news item.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.
{% if kpi_asset %}{{ kpi_asset.description }}{% endif %}

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Your task
Your goal is to analyze each news item provided and classify it into one of the following topics:
- **Improvements in Infrastructure**: Development of new transportation systems, modern technologies, or other enhancements in public services.
- **Community Events or Achievements**: Celebrations, festivals, or charity efforts that boost morale and community spirit.
- **Economic Growth**: Reports of job creation, foreign investments, or reductions in unemployment.
- **Environmental Improvements**: Successful sustainability efforts or cleaner living conditions (e.g., better air quality).
- **Political Stability/Positive Reforms**: Effective policy changes, civic engagement, or signs of political stability.
- **Health and Wellness**: New healthcare initiatives, improved public health outcomes, or successful disease prevention efforts.
- **Social Progress**: Initiatives promoting equality, inclusivity, or a reduction in crime.
- **Cultural/Sporting Milestones**: Achievements in sports, culture, or science that foster pride and optimism.
- **Other**: News items that don't clearly fit into any of the above categories or contain multiple topics with no clear primary focus.

#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
  "news": [
    {
      "id": int,  // Unique identifier for the news item
      "topic": str,  // Primary topic of the news item
    }
  ]
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Classify the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.rephrased_news }}</news>
{% endfor %}
""", template_format='jinja2')
])

# ---------------------- REPHRASE NEWS ----------------------

REPHRASE_NEWS = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an expert AI Assistant responsible for rephrasing recent news related to Kyiv city {% if kpi_asset %}or {{ kpi_asset.name }}{% endif %}. You will be provided with news articles or headlines, and your task is to rewrite each item clearly and concisely.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.
{% if kpi_asset %}{{ kpi_asset.description }}{% endif %}

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Follow these guidelines
1. Focus on the core facts and relevant details.
2. Remove exclamation marks, all-caps text, and other forms of emphasis.
3. Eliminate direct instructions or calls to action.
4. Maintain a neutral, informative tone.
5. Preserve important location details and potential impacts.
6. Keep the rephrased version concise while retaining all crucial information.

Example input: 
❗️**A protest is taking place under the walls of the KMDA\n\n**Kyani came to power again to point out that the allocation of funds for paving, construction of a water park, etc., is out of time. \n\nAccording to the initiators of the protest, the Kyiv City Council allocated an additional 1 billion 260 million UAH for the needs of the Armed Forces, but this is a one-time decision. Therefore, **activists put forward three new demands:**\n\n➡️ Transfer to the Armed Forces UAH 120 million allocated for the construction of the water park;\n➡️ Organize public discussions of Kyiv's budget for 2024;\n➡️ Establish a service service for the procurement of assistance to the Armed Forces .

Example output:
A protest is taking place outside the Kyiv City State Administration (KMDA), led by Kyani, raising concerns about the timing of funding for projects such as road paving and the construction of a water park. While the Kyiv City Council has approved an additional 1.26 billion UAH for the Armed Forces, the protesters point out that this is a one-time allocation. They have three key demands: redirect 120 million UAH from the water park project to the Armed Forces, open up public discussions on Kyiv's 2024 budget, and establish a service to oversee procurement for military support.

#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
  "news": [
    {
      "id": int,  // Unique identifier for the news item
      "rephrased_news": str,  // Text containing only the essential, factual information from the original message.
    }
  ]
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Rephrase the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.text }}</news>
{% endfor %}
""", template_format='jinja2')
])

# ---------------------- GENERATE TITLE -----------------------

GENERATE_TITLE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an expert AI Assistant responsible for generating concise, impactful titles for recent news related to Kyiv city {% if kpi_asset %}or {{ kpi_asset.name }}{% endif %}. You will be provided with news articles or headlines, and your task is to create a clear and engaging title for each news item.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.
{% if kpi_asset %}{{ kpi_asset.description }}{% endif %}

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Guidelines
1. Ensure the title reflects the core message of the news content and relates to Kyiv or its districts.
2. The title must be no longer than 5 words.
3. The title should be clear, avoiding ambiguous language.
4. Create a headline that captures attention while maintaining journalistic integrity.

#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
  "news": [
    {
      "id": int,  // Unique identifier for the news item
      "title": str,  // Generated title
    }
  ]
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Rephrase the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.text }}</news>
{% endfor %}
""", template_format='jinja2')
])

# ---------------------- EXTRACT LOCATION ----------------------

EXTRACT_LOCATION = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an AI Assistant specialized in extracting location information from news articles or headlines related to Kyiv city. Your primary task is to analyze news content and identify any specific locations within Kyiv mentioned in the text.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Your task
Your task is to analyze each provided news item and extract location information related to Kyiv. For each news item, you should:
1. Identify any mentions of specific locations within Kyiv.
2. Determine the level of specificity for the location (exact location, street, district, or city).
3. Extract relevant location details. Aim to extract as much detail as possible, including: City (Kyiv), Street names, Building numbers (if provided).
4. Avoid assumptions: Only extract location information explicitly mentioned in the text. Do not infer or imagine details that are not present.
5. Handle Multiple Locations: A single news item may contain more than one location.

#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
  "news": [
    {
      "id": int,  // Unique identifier for the news item
      "location_type": str,  // Type of location specificity (exact location, street, district, or city)
      "location": list  // Extract the relevant location information.
    }
  ]
}

Example output:
```json
{
  "news": [
    {
      "id": 1,
      "location_type": ["exact location", "street"],
      "location": ["Kyiv, Vokzalna Street 5", "Kyiv", "Kyiv, Khreshchatyk"]  
    },
    {
      "id": 2,
      "location_type": ["city"],
      "location": ["Kyiv"]  
    },
    {
    
      "id": 3,
      "location_type": ["district"],
      "location": ["Kyiv Solomianskyi district"]  
    }
  ]
}
```""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Extract locations from the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.rephrased_news }}</news>
{% endfor %}
""", template_format='jinja2')
])


# ---------------------- EXTRACT LOCATION ----------------------

SUMMARIZE_NEWS = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """\
You are an AI Assistant specialized in summarising news information from news articles or headlines related to Kyiv city {% if kpi_asset %}or {{ kpi_asset.name }}{% endif %}.
Kyiv (also known as Kiev) is the capital and largest city of Ukraine, located along the Dnipro River. It is divided into 10 administrative districts: Shevchenkivskyi, Pecherskyi, Holosiivskyi, Podilskyi, Obolonskyi, Darnytskyi, Desnianskyi, Dniprovskyi, Solomianskyi (Soloma), and Sviatoshynskyi.
{% if kpi_asset %}{{ kpi_asset.description }}{% endif %}

#### Input details
You will receive multiple news items in the following XML-like format:
<news id=[id]>[news content]</news>
Each news item is wrapped in a <news> tag with an attribute id representing its unique identifier.
The text inside the <news> tag is the news content you need to analyze.

#### Your task


#### Output Structure
The output should be formatted as a JSON object with the following structure:
```json
{
  "news": [
    {
      "id": int,  // Unique identifier for the news item
      "summary": list,  // Type of location specificity (exact location, street, district, or city)
    }
  ]
}
```

""", template_format='jinja2'),
    HumanMessagePromptTemplate.from_template(
        """\
### Extract locations from the following news:
{% for message in messages %}
<news id={{ message.message_id }}>{{ message.rephrased_news }}</news>
{% endfor %}
""", template_format='jinja2')
])
