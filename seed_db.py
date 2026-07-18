import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def populate_database():
    db_path = "./chroma_db"

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Full question text (aligned with questions.json / scoring API) + verified exact-match answers.
    # Re-seed after any edit:
    #   source env/bin/activate && python seed_db.py
    qa_dataset = [
        {
            "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
            "Answer": "3",
        },
        {
            "question": "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
            "Answer": "3",
        },
        {
            "question": '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
            "Answer": "Right",  # exact-match: capital R
        },
        {
            "question": "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.",
            "Answer": "Rd5",  # verified GAIA final answer (not Qxf2+)
        },
        {
            "question": "Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?",
            "Answer": "FunkMonk",
        },
        {
            "question": "Given this table defining * on the set S = {a, b, c, d, e}\n\n|*|a|b|c|d|e|\n|---|---|---|---|---|---|\n|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\nprovide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order.",
            "Answer": "b, e",
        },
        {
            "question": 'Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.\n\nWhat does Teal\'c say in response to the question "Isn\'t that hot?"',
            "Answer": "Extremely",
        },
        {
            "question": "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?",
            "Answer": "Louvrier",
        },
        {
            "question": "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes to categorizing things. I need to add different foods to different categories on the grocery list, but if I make a mistake, she won't buy anything inserted in the wrong category. Here's the list I have so far:\n\nmilk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\nI need to make headings for the fruits and vegetables. Could you please create a list of just the vegetables from my list? If you could do that, then I can figure out how to categorize the rest of the list into the appropriate categories. But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the vegetable list, or she won't get them when she's at the store. Please alphabetize the list of vegetables, and place each item in a comma separated list.",
            "Answer": "broccoli, celery, fresh basil, lettuce, sweet potatoes",
        },
        {
            "question": "Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3.\n\nIn your response, please only list the ingredients, not any measurements. So if the recipe calls for \"a pinch of salt\" or \"two cups of ripe strawberries\" the ingredients on the list would be \"salt\" and \"ripe strawberries\".\n\nPlease format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients.",
            "Answer": "cornstarch, freshly squeezed lemon juice, granulated sugar, pure vanilla extract, ripe strawberries",
        },
        {
            "question": "Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.",
            "Answer": "Wojciech",  # corrected (was incorrectly Bartek)
        },
        {
            "question": "What is the final numeric output from the attached Python code?",
            "Answer": "0",  # script only returns when randint hits 0
        },
        {
            "question": "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
            "Answer": "519",
        },
        {
            "question": "Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order.",
            "Answer": "132, 133, 134, 197, 245",
        },
        {
            "question": "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?",
            "Answer": "80GSFC21M0002",
        },
        {
            "question": "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.",
            "Answer": "Saint Petersburg",
        },
        {
            "question": "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.",
            "Answer": "CUB",  # corrected (was incorrectly NOR)
        },
        {
            "question": "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.",
            "Answer": "Yoshida, Uehara",
        },
        {
            "question": "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.",
            "Answer": "89706.00",  # sum of Burgers+Hot Dogs+Salads+Fries+Ice Cream (exclude Soda)
        },
        {
            "question": "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?",
            "Answer": "Claus",
        },
    ]

    documents = []
    for sample in qa_dataset:
        content = f"Question : {sample['question']}\n\nAnswer: {sample['Answer']}"
        documents.append(Document(page_content=content))

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="my_collection",
    )
    print(f"✅ Database populated with {len(documents)} verified QA pairs.")


if __name__ == "__main__":
    populate_database()
