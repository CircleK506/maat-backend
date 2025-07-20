# /maat/backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
import uuid
import os
from supabase import create_client, Client
import requests
from datetime import datetime
import json # Added for JSON parsing in strategy generation

app = FastAPI()

# ------------------ CORS Middleware ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Supabase Setup ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------ Pydantic Models ------------------
class Contact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    tags: List[str] = []
    source: Optional[str] = None
    status: Optional[str] = "active"

    class Config:
        from_attributes = True

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contact_id: Optional[str] = None
    message: str
    status: str = "pending"
    created_at: Optional[str] = None

    class Config:
        from_attributes = True

class Trigger(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    action_type: str
    target: str
    created_at: Optional[str] = None

    class Config:
        from_attributes = True

class DynamicForm(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    fields: List[Dict[str, Any]]
    created_at: Optional[str] = None

class FormSubmission(BaseModel):
    form_id: str
    contact_id: Optional[str] = None
    responses: Dict[str, Any]
    created_at: Optional[str] = None

class Campaign(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    status: str = "draft"
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    total_recipients: Optional[int] = 0
    open_rate: Optional[float] = 0.0
    click_rate: Optional[float] = 0.0
    conversions: Optional[int] = 0
    performance: Optional[Dict[str, Any]] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}

class Funnel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    status: str = "active"
    stages: Optional[List[Dict[str, Any]]] = []
    conversion_rate: Optional[float] = 0.0
    total_entries: Optional[int] = 0
    total_conversions: Optional[int] = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}

class Strategy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str
    priority: str = "Medium"
    category: Optional[str] = None
    details: Optional[str] = None
    generated_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}

# ------------------ LLM Integration (Groq) ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_if_not_set_in_env")
GROQ_MODEL = "llama3-70b-8192"

def ask_groq(prompt: str) -> str:
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_if_not_set_in_env":
        print("Warning: GROQ_API_KEY is not set. Groq integration will not work.")
        return "Groq API key not configured."

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Groq API: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Groq API: {e}")
    except KeyError:
        print("Unexpected response format from Groq API.")
        raise HTTPException(status_code=500, detail="Unexpected response format from Groq API")

# ------------------ Helper Functions ------------------
def check_and_trigger_actions(event_type: str, data: dict):
    """Check triggers for matching event_type and execute actions"""
    try:
        res = supabase.table("triggers").select("*").eq("event_type", event_type).execute()
        if res.data:
            for trigger in res.data:
                handle_trigger_action(trigger, data)
    except Exception as e:
        print(f"Error processing triggers: {e}")

def handle_trigger_action(trigger: dict, data: dict):
    """Execute the appropriate action based on trigger type"""
    try:
        if trigger["action_type"] == "send_email":
            # Create a message record
            message_data = {
                "contact_id": data.get("contact_id"),
                "message": f"Automated message for {trigger['target']}",
                "status": "pending"
            }
            supabase.table("messages").insert(message_data).execute()
            
        elif trigger["action_type"] == "tag_user" and data.get("contact_id"):
            # Add tag to contact
            contact = supabase.table("contacts").select("tags").eq("id", data["contact_id"]).execute().data
            if contact:
                current_tags = contact[0].get("tags", [])
                if trigger["target"] not in current_tags:
                    supabase.table("contacts").update({
                        "tags": current_tags + [trigger["target"]]
                    }).eq("id", data["contact_id"]).execute()
                    
        elif trigger["action_type"] == "ai_analysis":
            # Use AI to analyze and potentially take action
            prompt = f"Analyze this {trigger['event_type']} event with data: {data}. Recommended action for target '{trigger['target']}':"
            ai_response = ask_groq(prompt)
            print(f"AI Analysis Result: {ai_response}")
            
    except Exception as e:
        print(f"Error executing trigger action: {e}")

# ------------------ Contact Routes ------------------
@app.get("/api/contacts", response_model=List[Contact])
def get_contacts():
    try:
        res = supabase.table("contacts").select("*").execute()
        if res.data is None:
            return []
        return [Contact.model_validate(contact) for contact in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching contacts: {e}")

@app.post("/api/contacts", response_model=Contact)
def create_contact(contact: Contact):
    try:
        contact_data = contact.model_dump(exclude_unset=True)
        if 'id' in contact_data:
            del contact_data['id'] 

        res = supabase.table("contacts").insert(contact_data).execute()
        if res.data:
            # Trigger any new_contact events
            check_and_trigger_actions("new_contact", {"contact_id": res.data[0]["id"]})
            return Contact.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create contact in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating contact: {e}")

@app.delete("/api/contacts/{contact_id}")
def delete_contact(contact_id: str):
    try:
        res = supabase.table("contacts").delete().eq("id", contact_id).execute()
        if res.data and len(res.data) > 0:
            return {"message": f"Contact {contact_id} deleted successfully."}
        raise HTTPException(status_code=404, detail=f"Contact with ID {contact_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting contact: {e}")

# ------------------ Message Routes ------------------
@app.get("/api/messages", response_model=List[Message])
def get_messages(contact_id: Optional[str] = None):
    try:
        query = supabase.table("messages").select("*")
        if contact_id:
            query = query.eq("contact_id", contact_id)
        res = query.execute()
        return [Message.model_validate(msg) for msg in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {e}")

@app.post("/api/messages", response_model=Message)
def create_message(message: Message):
    try:
        message_data = message.model_dump(exclude_unset=True)
        if 'id' in message_data:
            del message_data['id']
        if 'created_at' not in message_data:
            message_data['created_at'] = datetime.now().isoformat()

        res = supabase.table("messages").insert(message_data).execute()
        if res.data:
            # Trigger message events
            check_and_trigger_actions("new_message", {
                "contact_id": message_data.get("contact_id"),
                "message_id": res.data[0]["id"]
            })
            return Message.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create message in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating message: {e}")

# ------------------ Trigger Routes ------------------
@app.get("/api/triggers", response_model=List[Trigger])
def get_triggers():
    try:
        res = supabase.table("triggers").select("*").execute()
        return [Trigger.model_validate(trigger) for trigger in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching triggers: {e}")

@app.post("/api/triggers", response_model=Trigger)
def create_trigger(trigger: Trigger):
    try:
        trigger_data = trigger.model_dump(exclude_unset=True)
        if 'id' in trigger_data:
            del trigger_data['id']
        if 'created_at' not in trigger_data:
            trigger_data['created_at'] = datetime.now().isoformat()

        res = supabase.table("triggers").insert(trigger_data).execute()
        if res.data:
            return Trigger.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create trigger in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating trigger: {e}")

# ------------------ Dynamic Form Routes ------------------
@app.get("/api/forms", response_model=List[DynamicForm])
def get_forms():
    try:
        res = supabase.table("forms").select("*").execute()
        return [DynamicForm.model_validate(form) for form in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching forms: {e}")

@app.post("/api/forms", response_model=DynamicForm)
def create_form(form: DynamicForm):
    try:
        form_data = form.model_dump(exclude_unset=True)
        if 'id' in form_data:
            del form_data['id']
        if 'created_at' not in form_data:
            form_data['created_at'] = datetime.now().isoformat()

        res = supabase.table("forms").insert(form_data).execute()
        if res.data:
            return DynamicForm.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create form in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating form: {e}")

@app.post("/api/form-submissions")
def submit_form(submission: FormSubmission):
    try:
        submission_data = submission.model_dump(exclude_unset=True)
        if 'created_at' not in submission_data:
            submission_data['created_at'] = datetime.now().isoformat()

        res = supabase.table("form_submissions").insert(submission_data).execute()
        if res.data:
            # Trigger form submission events
            check_and_trigger_actions("form_submission", {
                "form_id": submission_data["form_id"],
                "contact_id": submission_data.get("contact_id"),
                "responses": submission_data["responses"]
            })
            return {"message": "Form submitted successfully"}
        raise HTTPException(status_code=500, detail="Failed to submit form in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting form: {e}")

# ------------------ Campaign Routes (Supabase Integrated) ------------------
@app.get("/api/campaigns", response_model=List[Campaign])
def get_campaigns():
    """Retrieves all campaigns from the Supabase 'campaigns' table."""
    try:
        res = supabase.table("campaigns").select("*").execute()
        if res.data is None:
            return []
        return [Campaign.model_validate(campaign) for campaign in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching campaigns: {e}")

@app.post("/api/campaigns", response_model=Campaign)
def create_campaign(campaign: Campaign):
    """Creates a new campaign in the Supabase 'campaigns' table."""
    try:
        campaign_data = campaign.model_dump(exclude_unset=True)
        # Supabase will handle 'id' generation by default if not provided
        if 'id' in campaign_data:
            del campaign_data['id']
        # Supabase will handle 'created_at' and 'updated_at' defaults if not provided
        if 'created_at' in campaign_data:
            del campaign_data['created_at']
        if 'updated_at' in campaign_data:
            del campaign_data['updated_at']

        res = supabase.table("campaigns").insert(campaign_data).execute()
        if res.data:
            # Trigger campaign events after successful creation
            check_and_trigger_actions("campaign_created", {
                "campaign_id": res.data[0]["id"],
                "title": res.data[0]["title"],
                "status": res.data[0]["status"]
            })
            return Campaign.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create campaign in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating campaign: {e}")

@app.delete("/api/campaigns/{campaign_id}")
def delete_campaign(campaign_id: str):
    """Deletes a campaign from the Supabase 'campaigns' table by ID."""
    try:
        res = supabase.table("campaigns").delete().eq("id", campaign_id).execute()
        if res.data and len(res.data) > 0:
            return {"message": f"Campaign {campaign_id} deleted successfully."}
        raise HTTPException(status_code=404, detail=f"Campaign with ID {campaign_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting campaign: {e}")

@app.put("/api/campaigns/{campaign_id}", response_model=Campaign)
def update_campaign(campaign_id: str, campaign: Campaign):
    """Updates an existing campaign in the Supabase 'campaigns' table."""
    try:
        # Pydantic model_dump() includes only defined fields
        campaign_data = campaign.model_dump(exclude_unset=True)
        # Remove ID from update data, as it's used in .eq()
        if 'id' in campaign_data:
            del campaign_data['id']
        # Supabase handles updated_at, so remove if client provides it
        if 'updated_at' in campaign_data:
            del campaign_data['updated_at']
        
        res = supabase.table("campaigns").update(campaign_data).eq("id", campaign_id).execute()
        if res.data:
            return Campaign.model_validate(res.data[0])
        raise HTTPException(status_code=404, detail=f"Campaign with ID {campaign_id} not found or failed to update.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating campaign: {e}")

# ------------------ Funnel Routes (Supabase Integrated) ------------------
@app.get("/api/funnels", response_model=List[Funnel])
def get_funnels():
    """Retrieves all funnels from the Supabase 'funnels' table."""
    try:
        res = supabase.table("funnels").select("*").execute()
        if res.data is None:
            return []
        return [Funnel.model_validate(funnel) for funnel in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching funnels: {e}")

@app.post("/api/funnels", response_model=Funnel)
def create_funnel(funnel: Funnel):
    """Creates a new funnel in the Supabase 'funnels' table."""
    try:
        funnel_data = funnel.model_dump(exclude_unset=True)
        if 'id' in funnel_data:
            del funnel_data['id']
        if 'created_at' in funnel_data:
            del funnel_data['created_at']
        if 'updated_at' in funnel_data:
            del funnel_data['updated_at']

        res = supabase.table("funnels").insert(funnel_data).execute()
        if res.data:
            return Funnel.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create funnel in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating funnel: {e}")

@app.delete("/api/funnels/{funnel_id}")
def delete_funnel(funnel_id: str):
    """Deletes a funnel from the Supabase 'funnels' table by ID."""
    try:
        res = supabase.table("funnels").delete().eq("id", funnel_id).execute()
        if res.data and len(res.data) > 0:
            return {"message": f"Funnel {funnel_id} deleted successfully."}
        raise HTTPException(status_code=404, detail=f"Funnel with ID {funnel_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting funnel: {e}")

@app.put("/api/funnels/{funnel_id}", response_model=Funnel)
def update_funnel(funnel_id: str, funnel: Funnel):
    """Updates an existing funnel in the Supabase 'funnels' table."""
    try:
        funnel_data = funnel.model_dump(exclude_unset=True)
        if 'id' in funnel_data:
            del funnel_data['id']
        if 'updated_at' in funnel_data:
            del funnel_data['updated_at']
        
        res = supabase.table("funnels").update(funnel_data).eq("id", funnel_id).execute()
        if res.data:
            return Funnel.model_validate(res.data[0])
        raise HTTPException(status_code=404, detail=f"Funnel with ID {funnel_id} not found or failed to update.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating funnel: {e}")

# ------------------ Strategy Routes (Supabase Integrated & LLM) ------------------
@app.get("/api/strategy", response_model=List[Strategy])
def get_strategies(): # Renamed to plural for consistency
    """Retrieves all strategies from the Supabase 'strategies' table."""
    try:
        res = supabase.table("strategies").select("*").execute()
        if res.data is None:
            return []
        return [Strategy.model_validate(strategy) for strategy in res.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching strategies: {e}")

@app.post("/api/strategy", response_model=Strategy) # Changed endpoint from /api/strategy_generate
def create_strategy(strategy: Strategy):
    """Creates a new strategy in the Supabase 'strategies' table."""
    try:
        strategy_data = strategy.model_dump(exclude_unset=True)
        if 'id' in strategy_data:
            del strategy_data['id']
        if 'created_at' in strategy_data:
            del strategy_data['created_at']
        if 'updated_at' in strategy_data:
            del strategy_data['updated_at']

        res = supabase.table("strategies").insert(strategy_data).execute()
        if res.data:
            return Strategy.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to create strategy in Supabase.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating strategy: {e}")

@app.delete("/api/strategy/{strategy_id}")
def delete_strategy(strategy_id: str):
    """Deletes a strategy from the Supabase 'strategies' table by ID."""
    try:
        res = supabase.table("strategies").delete().eq("id", strategy_id).execute()
        if res.data and len(res.data) > 0:
            return {"message": f"Strategy {strategy_id} deleted successfully."}
        raise HTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting strategy: {e}")

@app.put("/api/strategy/{strategy_id}", response_model=Strategy)
def update_strategy(strategy_id: str, strategy: Strategy):
    """Updates an existing strategy in the Supabase 'strategies' table."""
    try:
        strategy_data = strategy.model_dump(exclude_unset=True)
        if 'id' in strategy_data:
            del strategy_data['id']
        if 'updated_at' in strategy_data:
            del strategy_data['updated_at']
        
        res = supabase.table("strategies").update(strategy_data).eq("id", strategy_id).execute()
        if res.data:
            return Strategy.model_validate(res.data[0])
        raise HTTPException(status_code=404, detail=f"Strategy with ID {strategy_id} not found or failed to update.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating strategy: {e}")

@app.post("/api/strategy/generate_and_save", response_model=Strategy) # New endpoint for generating AND saving
def generate_and_save_strategy(query: Dict[str, str]): # Expects {"input": "prompt"}
    if "input" not in query:
        raise HTTPException(status_code=400, detail="Missing 'input' in query body")
    
    # Use LLM to generate summary/details
    llm_prompt = f"Generate a short CRM strategy summary (max 100 words) and a more detailed plan based on the following: {query['input']}. Output in JSON format with 'summary', 'details', 'priority' (High/Medium/Low), and 'category'."
    ai_response_json_str = ask_groq(llm_prompt)
    
    try:
        # Attempt to parse the AI response as JSON
        ai_data = json.loads(ai_response_json_str)
        
        # Create a Strategy object from AI response
        new_strategy = Strategy(
            summary=ai_data.get("summary", "AI generated strategy"),
            details=ai_data.get("details"),
            priority=ai_data.get("priority", "Medium"),
            category=ai_data.get("category", "General"),
            generated_by="AI"
        )
        
        # Save the AI-generated strategy to Supabase
        strategy_data = new_strategy.model_dump(exclude_unset=True)
        if 'id' in strategy_data:
            del strategy_data['id']
        if 'created_at' in strategy_data:
            del strategy_data['created_at']
        if 'updated_at' in strategy_data:
            del strategy_data['updated_at']

        res = supabase.table("strategies").insert(strategy_data).execute()
        
        if res.data:
            return Strategy.model_validate(res.data[0])
        raise HTTPException(status_code=500, detail="Failed to save AI-generated strategy to Supabase.")
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI response was not valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating and saving strategy: {e}")


# ------------------ Webhook for External Events ------------------
@app.post("/api/webhook")
async def handle_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("event_type")
        data = payload.get("data", {})
        
        # Process the webhook and trigger any matching actions
        check_and_trigger_actions(event_type, data)
        
        return {"status": "processed"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing webhook: {e}")