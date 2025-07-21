from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any, Union
import uuid
import os
from supabase import create_client, Client
import requests
from datetime import datetime
import httpx
import json
from enum import Enum
from fastapi_utils.tasks import repeat_every

# --- VERY EARLY DEBUG ---
print("DEBUG: Application main.py started execution!")
print("APPLICATION STARTING UP", flush=True)
# --- END VERY EARLY DEBUG ---

app = FastAPI()

# ------------------ CORS Middleware ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Service Configuration ------------------
class EmailService(str, Enum):
    MAILGUN = "mailgun"
    MAILRELAY = "mailrelay"

# Load configuration from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN")
MAILRELAY_API_KEY = os.getenv("MAILRELAY_API_KEY")
MAILRELAY_GROUP_ID = os.getenv("MAILRELAY_GROUP_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
DEFAULT_EMAIL_SERVICE = os.getenv("DEFAULT_EMAIL_SERVICE", EmailService.MAILGUN)
DEFAULT_SENDER_NAME = os.getenv("DEFAULT_SENDER_NAME", "Your Company")

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

class ContactUpdate(BaseModel):
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    status: Optional[str] = None

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contact_id: Optional[str] = None
    campaign_id: Optional[str] = None
    message: str
    subject: Optional[str] = None
    status: str = "pending"
    service: Optional[EmailService] = None
    created_at: Optional[str] = None

class MessageUpdate(BaseModel):
    contact_id: Optional[str] = None
    campaign_id: Optional[str] = None
    message: Optional[str] = None
    subject: Optional[str] = None
    status: Optional[str] = None
    service: Optional[EmailService] = None

class Trigger(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    action_type: str
    target: str
    template_id: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    service: Optional[EmailService] = None
    created_at: Optional[str] = None

class TriggerUpdate(BaseModel):
    event_type: Optional[str] = None
    action_type: Optional[str] = None
    target: Optional[str] = None
    template_id: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    service: Optional[EmailService] = None

class Campaign(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    status: str = "draft"
    template_id: str
    audience: List[str] = []
    schedule_at: Optional[str] = None
    service: Optional[EmailService] = None
    created_at: Optional[str] = None

class CampaignUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    template_id: Optional[str] = None
    audience: Optional[List[str]] = None
    schedule_at: Optional[str] = None
    service: Optional[EmailService] = None

class EmailTemplate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    subject: str
    content: str
    variables: List[str] = []
    service: Optional[EmailService] = None
    created_at: Optional[str] = None

class EmailTemplateUpdate(BaseModel):
    name: Optional[str] = None
    subject: Optional[str] = None
    content: Optional[str] = None
    variables: Optional[List[str]] = None
    service: Optional[EmailService] = None

class Funnel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    stages: List[Dict[str, Any]] = []
    status: str = "active"
    created_at: Optional[str] = None

class FunnelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    stages: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None

class Strategy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    content: str
    category: Optional[str] = None
    status: str = "active"
    created_at: Optional[str] = None

class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None

class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class LLMResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int] = None

class PersonaChatRequest(BaseModel):
    message: str
    persona: str = "default"
    history: Optional[List[Dict[str, str]]] = None

# ------------------ Email Service Abstraction ------------------
class EmailServiceClient:
    @staticmethod
    async def send_email(service: EmailService, message_data: dict):
        if service == EmailService.MAILGUN:
            return await EmailServiceClient._send_via_mailgun(message_data)
        elif service == EmailService.MAILRELAY:
            return await EmailServiceClient._send_via_mailrelay(message_data)
        else:
            raise ValueError(f"Unsupported email service: {service}")

    @staticmethod
    async def _send_via_mailgun(message_data: dict):
        if not MAILGUN_API_KEY or not MAILGUN_DOMAIN:
            raise ValueError("Mailgun credentials not configured")
        
        contact = supabase.table("contacts").select("*").eq("id", message_data["contact_id"]).execute().data
        if not contact:
            raise ValueError("Contact not found")
        
        contact = contact[0]
        template = supabase.table("email_templates").select("*").eq("id", message_data.get("template_id")).execute().data
        if not template:
            raise ValueError("Template not found")
        
        template = template[0]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
                    auth=("api", MAILGUN_API_KEY),
                    data={
                        "from": f"{DEFAULT_SENDER_NAME} <mailgun@{MAILGUN_DOMAIN}>",
                        "to": contact["email"],
                        "subject": template["subject"],
                        "text": template["content"],
                        "html": template["content"]
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Mailgun API error details: {response.text if 'response' in locals() else 'No response'}")
            raise e

    @staticmethod
    async def _send_via_mailrelay(message_data: dict):
        if not MAILRELAY_API_KEY or not MAILRELAY_GROUP_ID:
            raise ValueError("Mailrelay credentials not configured")
        
        contact = supabase.table("contacts").select("*").eq("id", message_data["contact_id"]).execute().data
        if not contact:
            raise ValueError("Contact not found")
        
        contact = contact[0]
        template = supabase.table("email_templates").select("*").eq("id", message_data.get("template_id")).execute().data
        if not template:
            raise ValueError("Template not found")
        
        template = template[0]
        
        payload = {
            "email": contact["email"],
            "name": f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip(),
            "group_ids": [MAILRELAY_GROUP_ID],
            "subject": template["subject"],
            "html": template["content"]
        }
        
        headers = {
            "X-Auth-Token": MAILRELAY_API_KEY,
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                # IMPORTANT: Replace the IP address with a proper domain if Mailrelay provides one.
                # Using IP directly with HTTPS can cause SSL certificate validation issues.
                response = await client.post(
                    "https://ip-50-62-81-50.ip.secureserver.net/ccm/admin/api/version/2/",
                    headers=headers,
                    json=payload,
                    params={"type": "json"} # Moved from URL path to params
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Mailrelay API error details: {response.text if 'response' in locals() else 'No response'}")
            raise e

# ------------------ LLM Service Abstraction ------------------
class LLMService:
    @staticmethod
    async def generate_text(request: LLMRequest, background_tasks: BackgroundTasks = None):
        if GROQ_API_KEY:
            return await LLMService._groq_generate(request)
        elif OLLAMA_API_URL:
            return await LLMService._ollama_generate(request)
        else:
            raise HTTPException(status_code=400, detail="No LLM service configured")

    @staticmethod
    async def _groq_generate(request: LLMRequest):
        if not GROQ_API_KEY:
            raise HTTPException(status_code=400, detail="Groq API key not configured")
        
        model = request.model or "llama3-70b-8192"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": request.prompt}],
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    response=data['choices'][0]['message']['content'],
                    model=model,
                    tokens_used=data.get('usage', {}).get('total_tokens')
                )
        except Exception as e:
            if 'response' in locals():
                print(f"Groq API error details: {response.text}")
            raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    @staticmethod
    async def _ollama_generate(request: LLMRequest):
        model = request.model or "llama3"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_API_URL,
                    json={
                        "model": model,
                        "prompt": request.prompt,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_tokens
                        }
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                full_response = ""
                async for chunk in response.aiter_lines():
                    if chunk:
                        data = json.loads(chunk)
                        full_response += data.get("response", "")
                
                return LLMResponse(
                    response=full_response,
                    model=model
                )
        except Exception as e:
            if 'response' in locals():
                print(f"Ollama API error details: {response.text}")
            raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

# ------------------ Campaign Trigger Engine ------------------
async def process_campaign_triggers():
    """Process pending campaigns and send messages"""
    try:
        # Get campaigns ready to send
        now = datetime.now().isoformat()
        campaigns = supabase.table("campaigns").select("*").eq("status", "scheduled").lte("schedule_at", now).execute().data
        
        for campaign in campaigns:
            # Update campaign status
            supabase.table("campaigns").update({"status": "sending"}).eq("id", campaign["id"]).execute()
            
            # Get target audience
            audience = campaign.get("audience", [])
            if not audience:
                audience_filter = {}
            else:
                audience_filter = {"tags": {"cs": audience}}
            
            contacts = supabase.table("contacts").select("*").eq("status", "active").filter(**audience_filter).execute().data
            
            # Get template
            template = supabase.table("email_templates").select("*").eq("id", campaign["template_id"]).execute().data
            if not template:
                print(f"Template not found for campaign {campaign['id']}")
                continue
            
            template = template[0]
            
            # Create messages for each contact
            for contact in contacts:
                message_data = {
                    "contact_id": contact["id"],
                    "campaign_id": campaign["id"],
                    "subject": template["subject"],
                    "message": template["content"],
                    "status": "pending",
                    "service": campaign.get("service") or DEFAULT_EMAIL_SERVICE
                }
                
                supabase.table("messages").insert(message_data).execute()
            
            # Update campaign status
            supabase.table("campaigns").update({"status": "sent", "sent_at": now}).eq("id", campaign["id"]).execute()
            
    except Exception as e:
        print(f"Error processing campaign triggers: {str(e)}")

async def process_pending_messages():
    """Process pending messages in the queue"""
    try:
        messages = supabase.table("messages").select("*").eq("status", "pending").execute().data
        
        for message in messages:
            try:
                # Update status to processing
                supabase.table("messages").update({"status": "processing"}).eq("id", message["id"]).execute()
                
                # Send email
                service = message.get("service") or DEFAULT_EMAIL_SERVICE
                await EmailServiceClient.send_email(service, message)
                
                # Update status to sent
                supabase.table("messages").update({
                    "status": "sent",
                    "sent_at": datetime.now().isoformat()
                }).eq("id", message["id"]).execute()
                
            except Exception as e:
                print(f"Error sending message {message['id']}: {str(e)}")
                supabase.table("messages").update({
                    "status": "failed",
                    "error": str(e)
                }).eq("id", message["id"]).execute()
                
    except Exception as e:
        print(f"Error processing message queue: {str(e)}")

# ------------------ Contact Endpoints ------------------
@app.post("/api/contacts", response_model=Contact)
async def create_contact(contact: Contact):
    """Create a new contact"""
    contact_data = contact.model_dump(exclude_unset=True)
    if 'id' in contact_data:
        del contact_data['id']
    
    res = supabase.table("contacts").insert(contact_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create contact")
    
    return res.data[0]

@app.get("/api/contacts", response_model=List[Contact])
async def get_contacts():
    """Get all contacts"""
    res = supabase.table("contacts").select("*").execute()
    return res.data or []

@app.get("/api/contacts/{contact_id}", response_model=Contact)
async def get_contact(contact_id: str):
    """Get a single contact by ID"""
    res = supabase.table("contacts").select("*").eq("id", contact_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    return res.data[0]

@app.put("/api/contacts/{contact_id}", response_model=Contact)
async def update_contact(contact_id: str, contact_update: ContactUpdate):
    """Update an existing contact"""
    update_data = contact_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("contacts").update(update_data).eq("id", contact_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    return res.data[0]

@app.delete("/api/contacts/{contact_id}")
async def delete_contact(contact_id: str):
    """Delete a contact"""
    res = supabase.table("contacts").delete().eq("id", contact_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    return {"message": "Contact deleted successfully"}

# ------------------ Campaign Endpoints ------------------
@app.post("/api/campaigns", response_model=Campaign)
async def create_campaign(campaign: Campaign, background_tasks: BackgroundTasks):
    """Create a new campaign and schedule it for sending"""
    campaign_data = campaign.model_dump(exclude_unset=True)
    if 'id' in campaign_data:
        del campaign_data['id']
    
    res = supabase.table("campaigns").insert(campaign_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create campaign")
    
    created_campaign = res.data[0]
    
    # If campaign is scheduled for immediate sending
    if created_campaign["status"] == "scheduled" and not created_campaign.get("schedule_at"):
        background_tasks.add_task(process_campaign_triggers)
    
    return created_campaign

@app.get("/api/campaigns", response_model=List[Campaign])
async def get_campaigns():
    """Get all campaigns"""
    res = supabase.table("campaigns").select("*").execute()
    return res.data or []

@app.get("/api/campaigns/{campaign_id}", response_model=Campaign)
async def get_campaign(campaign_id: str):
    """Get a single campaign by ID"""
    res = supabase.table("campaigns").select("*").eq("id", campaign_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return res.data[0]

@app.put("/api/campaigns/{campaign_id}", response_model=Campaign)
async def update_campaign(campaign_id: str, campaign_update: CampaignUpdate):
    """Update an existing campaign"""
    update_data = campaign_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("campaigns").update(update_data).eq("id", campaign_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return res.data[0]

@app.delete("/api/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str):
    """Delete a campaign"""
    res = supabase.table("campaigns").delete().eq("id", campaign_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return {"message": "Campaign deleted successfully"}

# ------------------ Message Endpoints ------------------
@app.post("/api/messages/send", response_model=Message)
async def send_message(message: Message, background_tasks: BackgroundTasks):
    """Send a message immediately"""
    message_data = message.model_dump(exclude_unset=True)
    if 'id' in message_data:
        del message_data['id']
    
    message_data['status'] = "pending"
    
    res = supabase.table("messages").insert(message_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create message")
    
    background_tasks.add_task(process_pending_messages)
    return res.data[0]

@app.get("/api/messages", response_model=List[Message])
async def get_messages(contact_id: Optional[str] = None, campaign_id: Optional[str] = None):
    """Get all messages with optional filters"""
    query = supabase.table("messages").select("*")
    
    if contact_id:
        query = query.eq("contact_id", contact_id)
    if campaign_id:
        query = query.eq("campaign_id", campaign_id)
    
    res = query.execute()
    return res.data or []

@app.get("/api/messages/{message_id}", response_model=Message)
async def get_message(message_id: str):
    """Get a single message by ID"""
    res = supabase.table("messages").select("*").eq("id", message_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return res.data[0]

@app.put("/api/messages/{message_id}", response_model=Message)
async def update_message(message_id: str, message_update: MessageUpdate):
    """Update an existing message"""
    update_data = message_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("messages").update(update_data).eq("id", message_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return res.data[0]

@app.delete("/api/messages/{message_id}")
async def delete_message(message_id: str):
    """Delete a message"""
    res = supabase.table("messages").delete().eq("id", message_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {"message": "Message deleted successfully"}

# ------------------ Trigger Endpoints ------------------
@app.post("/api/triggers", response_model=Trigger)
async def create_trigger(trigger: Trigger):
    """Create a new trigger"""
    trigger_data = trigger.model_dump(exclude_unset=True)
    if 'id' in trigger_data:
        del trigger_data['id']
    
    res = supabase.table("triggers").insert(trigger_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create trigger")
    
    return res.data[0]

@app.get("/api/triggers", response_model=List[Trigger])
async def get_triggers():
    """Get all triggers"""
    res = supabase.table("triggers").select("*").execute()
    return res.data or []

@app.get("/api/triggers/{trigger_id}", response_model=Trigger)
async def get_trigger(trigger_id: str):
    """Get a single trigger by ID"""
    res = supabase.table("triggers").select("*").eq("id", trigger_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Trigger not found")
    
    return res.data[0]

@app.put("/api/triggers/{trigger_id}", response_model=Trigger)
async def update_trigger(trigger_id: str, trigger_update: TriggerUpdate):
    """Update an existing trigger"""
    update_data = trigger_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("triggers").update(update_data).eq("id", trigger_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Trigger not found")
    
    return res.data[0]

@app.delete("/api/triggers/{trigger_id}")
async def delete_trigger(trigger_id: str):
    """Delete a trigger"""
    res = supabase.table("triggers").delete().eq("id", trigger_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Trigger not found")
    
    return {"message": "Trigger deleted successfully"}

# ------------------ Email Template Endpoints ------------------
@app.post("/api/email-templates", response_model=EmailTemplate)
async def create_email_template(template: EmailTemplate):
    """Create a new email template"""
    template_data = template.model_dump(exclude_unset=True)
    if 'id' in template_data:
        del template_data['id']
    
    res = supabase.table("email_templates").insert(template_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create email template")
    
    return res.data[0]

@app.get("/api/email-templates", response_model=List[EmailTemplate])
async def get_email_templates():
    """Get all email templates"""
    res = supabase.table("email_templates").select("*").execute()
    return res.data or []

@app.get("/api/email-templates/{template_id}", response_model=EmailTemplate)
async def get_email_template(template_id: str):
    """Get a single email template by ID"""
    res = supabase.table("email_templates").select("*").eq("id", template_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Email template not found")
    
    return res.data[0]

@app.put("/api/email-templates/{template_id}", response_model=EmailTemplate)
async def update_email_template(template_id: str, template_update: EmailTemplateUpdate):
    """Update an existing email template"""
    update_data = template_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("email_templates").update(update_data).eq("id", template_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Email template not found")
    
    return res.data[0]

@app.delete("/api/email-templates/{template_id}")
async def delete_email_template(template_id: str):
    """Delete an email template"""
    res = supabase.table("email_templates").delete().eq("id", template_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Email template not found")
    
    return {"message": "Email template deleted successfully"}

# ------------------ Funnel Endpoints ------------------
@app.post("/api/funnels", response_model=Funnel)
async def create_funnel(funnel: Funnel):
    """Create a new funnel"""
    funnel_data = funnel.model_dump(exclude_unset=True)
    if 'id' in funnel_data:
        del funnel_data['id']
    
    res = supabase.table("funnels").insert(funnel_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create funnel")
    
    return res.data[0]

@app.get("/api/funnels", response_model=List[Funnel])
async def get_funnels():
    """Get all funnels"""
    res = supabase.table("funnels").select("*").execute()
    return res.data or []

@app.get("/api/funnels/{funnel_id}", response_model=Funnel)
async def get_funnel(funnel_id: str):
    """Get a single funnel by ID"""
    res = supabase.table("funnels").select("*").eq("id", funnel_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Funnel not found")
    
    return res.data[0]

@app.put("/api/funnels/{funnel_id}", response_model=Funnel)
async def update_funnel(funnel_id: str, funnel_update: FunnelUpdate):
    """Update an existing funnel"""
    update_data = funnel_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("funnels").update(update_data).eq("id", funnel_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Funnel not found")
    
    return res.data[0]

@app.delete("/api/funnels/{funnel_id}")
async def delete_funnel(funnel_id: str):
    """Delete a funnel"""
    res = supabase.table("funnels").delete().eq("id", funnel_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Funnel not found")
    
    return {"message": "Funnel deleted successfully"}

# ------------------ Strategy Endpoints ------------------
@app.post("/api/strategy", response_model=Strategy)
async def create_strategy(strategy: Strategy):
    """Create a new strategy (manual input)"""
    strategy_data = strategy.model_dump(exclude_unset=True)
    if 'id' in strategy_data:
        del strategy_data['id']
    
    res = supabase.table("strategies").insert(strategy_data).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create strategy")
    
    return res.data[0]

@app.get("/api/strategy", response_model=List[Strategy])
async def get_strategies():
    """Get all strategies"""
    res = supabase.table("strategies").select("*").execute()
    return res.data or []

@app.get("/api/strategy/{strategy_id}", response_model=Strategy)
async def get_strategy(strategy_id: str):
    """Get a single strategy by ID"""
    res = supabase.table("strategies").select("*").eq("id", strategy_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return res.data[0]

@app.put("/api/strategy/{strategy_id}", response_model=Strategy)
async def update_strategy(strategy_id: str, strategy_update: StrategyUpdate):
    """Update an existing strategy"""
    update_data = strategy_update.model_dump(exclude_unset=True, exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    res = supabase.table("strategies").update(update_data).eq("id", strategy_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return res.data[0]

@app.delete("/api/strategy/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """Delete a strategy"""
    res = supabase.table("strategies").delete().eq("id", strategy_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    return {"message": "Strategy deleted successfully"}

# ------------------ LLM Endpoints ------------------
@app.post("/api/llm/generate", response_model=LLMResponse)
async def generate_text(request: LLMRequest):
    """Generate text using configured LLM service"""
    return await LLMService.generate_text(request)

@app.post("/api/llm/persona-chat", response_model=LLMResponse)
async def persona_chat(request: PersonaChatRequest):
    """Chat with a specific persona using LLM"""
    prompt = f"""You are a {request.persona} assistant. Respond appropriately to the following message:
    
    Message: {request.message}
    
    Previous conversation history:
    {json.dumps(request.history or [])}
    """
    
    llm_request = LLMRequest(
        prompt=prompt,
        model="llama3-70b-8192" if GROQ_API_KEY else "llama3",
        temperature=0.7
    )
    
    return await LLMService.generate_text(llm_request)

@app.post("/api/strategy_generate", response_model=LLMResponse)
async def run_strategy_generation(request: LLMRequest):
    """Generate marketing strategy using LLM"""
    # This endpoint can use the LLMService to generate a strategy based on the prompt
    # and then optionally save it to the strategies table.
    try:
        llm_response = await LLMService.generate_text(request)
        
        # Example of saving generated strategy (you might need to parse/structure this more)
        # Assuming the LLM response contains a structured strategy, e.g., JSON
        # For simplicity, here we'll just take the response content.
        
        # You'll likely need a more sophisticated parsing of the LLM response
        # to fit it into your 'Strategy' model if it's not directly structured.
        
        # For demonstration, let's assume LLM response content can be directly used as 'content'
        new_strategy = Strategy(
            name=f"AI Generated Strategy - {datetime.now().strftime('%Y%m%d%H%M%S')}",
            description="Generated by AI based on prompt",
            content=llm_response.response,
            category="AI-Generated",
            status="draft" # Or 'active', depending on your workflow
        )
        
        # Save the AI-generated strategy to Supabase
        strategy_data = new_strategy.model_dump(exclude_unset=True)
        # Assuming Supabase handles 'id' and 'created_at' automatically on insert
        if 'id' in strategy_data:
            del strategy_data['id']
        if 'created_at' in strategy_data:
            del strategy_data['created_at']
        if 'updated_at' in strategy_data: # If you have an updated_at column
            del strategy_data['updated_at']

        res = supabase.table("strategies").insert(strategy_data).execute()
        
        if res.data:
            return LLMResponse(response=res.data[0]['content'], model=llm_response.model, tokens_used=llm_response.tokens_used)
        raise HTTPException(status_code=500, detail="Failed to save AI-generated strategy to Supabase.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating and saving strategy: {e}")

# ------------------ Webhook for External Events ------------------
@app.post("/api/webhook")
async def handle_webhook(request: Request):
    try:
        payload = await request.json()
        event_type = payload.get("event_type")
        data = payload.get("data", {})
        
        # TODO: Implement a system to process webhooks and trigger corresponding actions
        # For example, if event_type is 'new_lead', create a contact and enroll in a funnel
        
        print(f"Received webhook: Type='{event_type}', Data={data}")
        
        # Example: if event_type == "contact_created":
        #     contact = Contact(**data)
        #     supabase.table("contacts").insert(contact.model_dump()).execute()
        
        return {"status": "processed", "received_event_type": event_type}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing webhook: {e}")

# ------------------ Periodic Tasks ------------------
@app.on_event("startup")
@repeat_every(seconds=60 * 5) # Run every 5 minutes
async def run_periodic_tasks():
    await process_campaign_triggers()
    await process_pending_messages()
