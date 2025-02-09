from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr, BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import jinja2
from datetime import datetime

class EmailSchema(BaseModel):
    """Email message schema."""
    recipients: List[EmailStr]
    subject: str
    body: Optional[str] = None
    template_name: Optional[str] = None
    template_data: Optional[Dict] = None
    created_at: str = "2025-02-09 09:56:49"
    processed_by: str = "kaxm23"

# Email configuration
conf = ConnectionConfig(
    MAIL_USERNAME = "your-email@example.com",
    MAIL_PASSWORD = "your-email-password",
    MAIL_FROM = "your-email@example.com",
    MAIL_PORT = 587,
    MAIL_SERVER = "smtp.gmail.com",
    MAIL_FROM_NAME = "Your App Name",
    MAIL_TLS = True,
    MAIL_SSL = False,
    USE_CREDENTIALS = True,
    TEMPLATE_FOLDER = Path(__file__).parent / 'email_templates'
)

# Initialize FastMail
fastmail = FastMail(conf)

async def send_email(email: EmailSchema) -> Dict:
    """
    Send email using FastMail.
    
    Args:
        email: Email message data
        
    Returns:
        Dict: Send result
    """
    try:
        # Create message
        if email.template_name:
            # Use template
            template = _get_template(email.template_name)
            html = template.render(**(email.template_data or {}))
        else:
            # Use raw body
            html = email.body or ""
        
        message = MessageSchema(
            subject=email.subject,
            recipients=email.recipients,
            body=html,
            subtype="html"
        )
        
        # Send email
        await fastmail.send_message(message)
        
        return {
            "status": "success",
            "message": "Email sent successfully",
            "recipients": len(email.recipients),
            "timestamp": "2025-02-09 09:56:49",
            "processed_by": "kaxm23"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to send email: {str(e)}",
            "timestamp": "2025-02-09 09:56:49",
            "processed_by": "kaxm23"
        }

def _get_template(template_name: str) -> jinja2.Template:
    """Get email template."""
    template_loader = jinja2.FileSystemLoader(conf.TEMPLATE_FOLDER)
    template_env = jinja2.Environment(loader=template_loader)
    return template_env.get_template(template_name)