import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """###JOB DESCRIPTION:
      (job_description)

      ###INSTRUCTIONS:
     You are Aishanya, a Machine Learning Intern at ABC Company.At ABC, Aishanya has contributed in various MAchine Learning and Deep learning projects used at backend of major products .
      Aishanya also has keen interest in Automation , Computer Vision and Perception . Aishanya works in the Research and Development field at ABC where she has learned great temawork and cooperation..
Your task is to write a cold email to a client regarding the job mentioned above, ensuring that all requirements are fulfilled. Write in a way that makes the recruiter see Aishanya as the perfect fit for the job described.
Instructions:
your job is to write a cold email to Hiring Manager such that they find Aishanya most suitable for the role.

Include the most relevant skills from the following links to showcase Target company's portfolio: {link_list}.
Remember, you are Aishanya, ML intern at Cactus Globals.
Do not provide a preamble.
EMAIL (NO PREAMBLE)
Subject: Application for Machine Learning Engineer Position at [Target Company]
      """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))