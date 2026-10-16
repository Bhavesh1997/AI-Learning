"""
02_structured_outputs.py
------------------------
Getting reliable, typed data out of an LLM using Pydantic + OpenAI Structured Outputs.

Run from the project root:
    python "LLM Fundamental/02_structured_outputs.py"

Each demo below is a function. At the bottom, the DEMOS list controls which ones run.
Comment a line out with Ctrl + / to skip that demo (each online demo makes one API call).
"""

import json
from enum import Enum
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o"  # must be a model that supports Structured Outputs


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_refusal(result):
    """Return the refusal text if the model refused the request, else None."""
    for item in result.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    return part.refusal
    return None


def safe_parse(**kwargs):
    """
    Call responses.parse and handle the ways it can fail quietly:
      - the model refuses            -> output_parsed is None
      - the response is incomplete   -> e.g. cut off by a token limit
    Returns the parsed Pydantic object, or None.
    """
    result = client.responses.parse(model=MODEL, **kwargs)

    refusal = get_refusal(result)
    if refusal:
        print("Model refused:", refusal)
        return None

    if result.status != "completed":
        reason = result.incomplete_details.reason if result.incomplete_details else "unknown"
        print(f"Response not completed: {result.status} ({reason})")
        return None

    if result.usage:
        u = result.usage
        print(f"[tokens] in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")

    return result.output_parsed


# ----------------------------------------------------------------------------
# 0. What actually gets sent? (offline, no API call)
# ----------------------------------------------------------------------------
class Person(BaseModel):
    name: str
    age: int
    city: str


def demo_inspect_schema():
    """Pydantic turns your class into a JSON Schema. That schema is what constrains the model."""
    print(json.dumps(Person.model_json_schema(), indent=2))
    print(
        "\nThe SDK also marks every field as required and sets additionalProperties=false\n"
        "before sending it. The model can only produce JSON that fits this shape."
    )


# ----------------------------------------------------------------------------
# 1. The basic pattern
# ----------------------------------------------------------------------------
def demo_basic():
    result = client.responses.parse(
        model=MODEL,
        instructions="You are a helpful assistant.",
        input="Give me the name, age and city of a fictional person.",
        text_format=Person,  # <- the schema
    )

    person = result.output_parsed  # a real Person object, not text
    print(person)
    print(person.name, person.age, person.city)

    print("\nType of person.age:", type(person.age).__name__)
    print("Age next year:", person.age + 1)  # works because age is a real int

    print("\nAs a dict:", person.model_dump())
    print("As JSON:\n" + person.model_dump_json(indent=2))


# ----------------------------------------------------------------------------
# 2. Field descriptions guide the model
# ----------------------------------------------------------------------------
class PersonProfile(BaseModel):
    full_name: str = Field(description="First and last name")
    age: int = Field(description="Age in whole years")
    city: str = Field(description="City of residence")
    bio: str = Field(description="One-sentence bio, under 25 words")


def demo_descriptions():
    profile = safe_parse(
        instructions="You generate realistic fictional people.",
        input="Create a fictional person who lives in India.",
        text_format=PersonProfile,
    )
    if profile:
        print(profile.model_dump_json(indent=2))


# ----------------------------------------------------------------------------
# 3. Richer types: Enum, list, nested model, optional field
# ----------------------------------------------------------------------------
class Occupation(str, Enum):
    ENGINEER = "engineer"
    DOCTOR = "doctor"
    TEACHER = "teacher"
    OTHER = "other"


class Address(BaseModel):
    city: str
    state: str
    country: str


class RichPerson(BaseModel):
    name: str
    age: int
    occupation: Occupation  # restricted to the four values above
    hobbies: list[str] = Field(description="3 to 5 hobbies")
    languages: list[str] = Field(description="Languages the person speaks")
    address: Address  # nested object
    nickname: str | None = Field(description="Nickname if they have one, otherwise null")


def demo_rich_types():
    person = safe_parse(
        instructions="You generate realistic fictional people.",
        input="Create a fictional person living in Pune.",
        text_format=RichPerson,
    )
    if not person:
        return

    print(person.model_dump_json(indent=2))
    print("\nOccupation value:", person.occupation.value)  # Enum -> plain string
    print("City:", person.address.city)  # nested access
    for hobby in person.hobbies:
        print(" -", hobby)
    print("Has nickname?", person.nickname is not None)


# ----------------------------------------------------------------------------
# 4. Extraction from messy text (and not letting the model guess)
# ----------------------------------------------------------------------------
JOB_POST = """
Senior Data Scientist - Pharma Analytics
Acme Analytics, Pune (hybrid, 3 days in office)

We are looking for someone with 5+ years of experience in Python, SQL and
machine learning. Experience with LangGraph or RAG pipelines is a plus.
Compensation is competitive and will be discussed during the interview process.
"""


class JobPosting(BaseModel):
    title: str
    company: str | None
    location: str | None
    work_mode: Literal["onsite", "hybrid", "remote"] | None
    min_years_experience: int | None
    skills: list[str]
    salary_min: int | None = Field(description="Only if a number is stated, otherwise null")
    salary_max: int | None = Field(description="Only if a number is stated, otherwise null")


def demo_extraction():
    job = safe_parse(
        instructions=(
            "Extract job details from the text. Use only what is stated. "
            "If a field is not stated, return null. Never guess."
        ),
        input=JOB_POST,
        text_format=JobPosting,
        temperature=0,  # extraction should be deterministic
    )
    if not job:
        return

    print(job.model_dump_json(indent=2))
    missing = [field for field, value in job.model_dump().items() if value is None]
    print("\nFields the model correctly left empty:", missing)
    print("Expected: salary_min and salary_max are null, because no number was given.")


# ----------------------------------------------------------------------------
# 5. Classification over a batch of inputs
# ----------------------------------------------------------------------------
class Ticket(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    sentiment: Literal["positive", "neutral", "negative"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(description="One sentence, maximum 15 words")
    needs_human: bool = Field(description="True if a support agent should reply personally")


TICKETS = [
    "I was charged twice for my subscription this month and nobody replies to my emails!",
    "How do I change the email address on my account?",
    "The export button crashes the app every time. We have a client demo in one hour.",
    "Just wanted to say the new dashboard is great, thanks!",
]


def demo_classification():
    print(f"{'category':<10} {'sentiment':<9} {'priority':<8} human  summary")
    print("-" * 78)
    for text in TICKETS:
        ticket = safe_parse(
            instructions="You triage customer support tickets.",
            input=text,
            text_format=Ticket,
            temperature=0,
        )
        if ticket:
            print(
                f"{ticket.category:<10} {ticket.sentiment:<9} {ticket.priority:<8} "
                f"{str(ticket.needs_human):<6} {ticket.summary}"
            )


# ----------------------------------------------------------------------------
# 6. Lists of objects (the root must be an object, so wrap the list)
# ----------------------------------------------------------------------------
class ActionItem(BaseModel):
    task: str
    owner: str | None = Field(description="Person responsible, or null if not stated")
    due: str | None = Field(description="Deadline as stated, e.g. 'Friday', or null")


class MeetingNotes(BaseModel):
    summary: str = Field(description="Two-sentence summary of the meeting")
    decisions: list[str]
    action_items: list[ActionItem]  # a list inside the root object


TRANSCRIPT = """
Priya: Let's go with the vendor B quote, it's cheaper and the timelines match.
Rahul: Agreed. I'll send the signed agreement by Friday.
Priya: Great. Someone also needs to update the project plan.
Rahul: Meera can do that, maybe early next week?
Meera: Sure, I'll take it. Also, we still need to book the kickoff room.
"""


def demo_list_of_objects():
    # A top-level list is not allowed, so a wrapper model with a list field is used.
    notes = safe_parse(
        instructions="Turn the meeting transcript into structured notes.",
        input=TRANSCRIPT,
        text_format=MeetingNotes,
        temperature=0,
    )
    if not notes:
        return

    print("Summary:", notes.summary)
    print("\nDecisions:")
    for d in notes.decisions:
        print(" -", d)
    print("\nAction items:")
    for item in notes.action_items:
        print(f" - {item.task} | owner: {item.owner} | due: {item.due}")


# ----------------------------------------------------------------------------
# 7. Field ORDER matters: put reasoning before the answer
# ----------------------------------------------------------------------------
class MathAnswer(BaseModel):
    # The model writes fields in order, so it "thinks" in steps before committing to the answer.
    steps: list[str] = Field(description="Work through the problem step by step")
    final_answer: float
    unit: str


def demo_reasoning_first():
    answer = safe_parse(
        instructions="Solve the problem carefully.",
        input=(
            "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. "
            "What is its average speed for the whole trip?"
        ),
        text_format=MathAnswer,
        temperature=0,
    )
    if not answer:
        return

    for i, step in enumerate(answer.steps, start=1):
        print(f"Step {i}: {step}")
    print(f"\nAnswer: {answer.final_answer} {answer.unit}  (expected 75.0)")
    print("Tip: if final_answer came first, the model would have to guess before reasoning.")


# ----------------------------------------------------------------------------
# 8. Local validation and retry
# The schema guarantees SHAPE, not sensible values. Pydantic validators check values.
# ----------------------------------------------------------------------------
class CheckedPerson(BaseModel):
    name: str
    age: int
    city: str

    @field_validator("age")
    @classmethod
    def age_must_be_realistic(cls, v: int) -> int:
        if not 0 <= v <= 120:
            raise ValueError("age must be between 0 and 120")
        return v

    @field_validator("name", "city")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


def extract_with_retry(prompt: str, max_attempts: int = 3):
    """Parse, and if local validation fails, tell the model what was wrong and try again."""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, max_attempts + 1):
        try:
            result = client.responses.parse(
                model=MODEL,
                input=messages,
                text_format=CheckedPerson,
            )
        except ValidationError as err:
            print(f"Attempt {attempt}: validation failed -> {err.error_count()} problem(s)")
            messages.append(
                {
                    "role": "user",
                    "content": f"Your previous answer failed validation:\n{err}\nReturn a corrected answer.",
                }
            )
            continue

        if get_refusal(result):
            print("Model refused.")
            return None

        print(f"Attempt {attempt}: OK")
        return result.output_parsed

    print("Gave up after", max_attempts, "attempts.")
    return None


def demo_validation_and_retry():
    # Part A (offline): see Pydantic reject bad data
    print("Part A: validating a bad dict locally")
    try:
        CheckedPerson(name="Asha", age=250, city="   ")
    except ValidationError as err:
        for problem in err.errors():
            print(f" - field {problem['loc'][0]}: {problem['msg']}")

    # Part B (online): ask for something likely to break the age rule, with retry
    print("\nPart B: retry loop")
    person = extract_with_retry("Give me a fictional person who is 150 years old, with a city.")
    print("Final result:", person)


# ----------------------------------------------------------------------------
# 9. The older Chat Completions API (you will see this in many tutorials)
# ----------------------------------------------------------------------------
def demo_chat_completions_version():
    completion = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me the name, age and city of a fictional person."},
        ],
        response_format=Person,  # <- named response_format here, text_format in Responses API
    )

    message = completion.choices[0].message
    if message.refusal:
        print("Model refused:", message.refusal)
    else:
        print(message.parsed)  # <- .parsed here, .output_parsed in Responses API


# ----------------------------------------------------------------------------
# Run the demos. Comment out any line (Ctrl + /) to skip it.
# ----------------------------------------------------------------------------
DEMOS = [
    demo_inspect_schema,  # offline
    demo_basic,
    demo_descriptions,
    demo_rich_types,
    demo_extraction,
    demo_classification,
    demo_list_of_objects,
    demo_reasoning_first,
    demo_validation_and_retry,
    demo_chat_completions_version,
]


def main():
    for demo in DEMOS:
        header(demo.__name__)
        try:
            demo()
        except Exception as err:  # one failing demo should not stop the rest
            print(f"Demo failed: {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()