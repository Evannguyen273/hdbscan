import logging
import json
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
import traceback

class PipelineLogger:
    """Enhanced logging for pipeline operations"""
    
    def __init__(self, pipeline_name: str, tech_center: Optional[str] = None):
        self.pipeline_name = pipeline_name
        self.tech_center = tech_center
        self.start_time = datetime.now()
        
        # Setup logging format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        
    def log_stage_start(self, stage_name: str, details: Optional[Dict] = None):
        """Log the start of a pipeline stage"""
        message = f"[{self.pipeline_name}] Starting stage: {stage_name}"
        if self.tech_center:
            message += f" | Tech Center: {self.tech_center}"
        
        logging.info(message)
        
        if details:
            logging.info(f"Stage details: {json.dumps(details, indent=2)}")
    
    def log_stage_complete(self, stage_name: str, metrics: Optional[Dict] = None):
        """Log successful completion of a pipeline stage"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        message = f"[{self.pipeline_name}] Completed stage: {stage_name} | Duration: {duration:.2f}s"
        if self.tech_center:
            message += f" | Tech Center: {self.tech_center}"
            
        logging.info(message)
        
        if metrics:
            logging.info(f"Stage metrics: {json.dumps(metrics, indent=2)}")
    
    def log_progress(self, current: int, total: int, item_name: str = "items"):
        """Log progress during processing"""
        percentage = (current / total) * 100
        message = f"[{self.pipeline_name}] Progress: {current}/{total} {item_name} ({percentage:.1f}%)"
        
        if self.tech_center:
            message += f" | Tech Center: {self.tech_center}"
            
        logging.info(message)
    
    def log_error(self, stage_name: str, error: Exception, context: Optional[Dict] = None):
        """Log detailed error information"""
        error_data = {
            "pipeline": self.pipeline_name,
            "tech_center": self.tech_center,
            "stage": stage_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.error(f"PIPELINE_ERROR: {json.dumps(error_data, indent=2)}")
        return error_data

class ErrorNotifier:
    """Handle error notifications via multiple channels"""
    
    def __init__(self, config):
        self.config = config
        
    def send_error_notification(self, error_details: Dict[str, Any]):
        """Send error notification via configured channels"""
        
        # Email notification
        if hasattr(self.config, 'notifications') and self.config.notifications.get('email', {}).get('enabled', False):
            self._send_email(error_details)
        
        # Teams webhook notification
        if hasattr(self.config, 'notifications') and self.config.notifications.get('teams', {}).get('enabled', False):
            self._send_teams_notification(error_details)
        
        # Console notification (always enabled)
        self._send_console_notification(error_details)
    
    def _send_console_notification(self, error_details: Dict[str, Any]):
        """Send console notification"""
        print(f"\n🚨 PIPELINE ERROR ALERT 🚨")
        print(f"Pipeline: {error_details['pipeline']}")
        print(f"Tech Center: {error_details.get('tech_center', 'All')}")
        print(f"Stage: {error_details['stage']}")
        print(f"Error: {error_details['error']}")
        print(f"Time: {error_details['timestamp']}")
        print("=" * 50)
    
    def _send_email(self, error_details: Dict[str, Any]):
        """Send email notification"""
        try:
            subject = f"🚨 HDBSCAN Pipeline Error - {error_details['pipeline']}"
            
            body = f"""
            HDBSCAN Clustering Pipeline Error Report
            
            Pipeline: {error_details['pipeline']}
            Tech Center: {error_details.get('tech_center', 'All')}
            Stage: {error_details['stage']}
            Error Type: {error_details.get('error_type', 'Unknown')}
            Error Message: {error_details['error']}
            Timestamp: {error_details['timestamp']}
            
            Context: {json.dumps(error_details.get('context', {}), indent=2)}
            
            Please check the pipeline logs for detailed traceback information.
            
            Automated notification from HDBSCAN Pipeline System
            """
            
            msg = MIMEMultipart()
            msg['From'] = self.config.notifications.email.get('from', 'pipeline@company.com')
            msg['To'] = ', '.join(self.config.notifications.email.get('recipients', ['admin@company.com']))
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (configure SMTP settings)
            smtp_server = self.config.notifications.email.get('smtp_server', 'smtp.company.com')
            smtp_port = self.config.notifications.email.get('smtp_port', 587)
            
            print(f"📧 Email notification prepared (configure SMTP to send)")
            print(f"To: {msg['To']}")
            print(f"Subject: {subject}")
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
    
    def _send_teams_notification(self, error_details: Dict[str, Any]):
        """Send Teams webhook notification"""
        try:
            webhook_url = self.config.notifications.teams.get('webhook_url')
            if not webhook_url:
                return
                
            teams_message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": f"Pipeline Error: {error_details['pipeline']}",
                "themeColor": "FF0000",
                "sections": [{
                    "activityTitle": "🚨 HDBSCAN Pipeline Error",
                    "activitySubtitle": f"Pipeline: {error_details['pipeline']}",
                    "facts": [
                        {"name": "Tech Center", "value": error_details.get('tech_center', 'All')},
                        {"name": "Stage", "value": error_details['stage']},
                        {"name": "Error", "value": error_details['error'][:100] + "..." if len(error_details['error']) > 100 else error_details['error']},
                        {"name": "Time", "value": error_details['timestamp']}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=teams_message)
            if response.status_code == 200:
                logging.info("Teams notification sent successfully")
            else:
                logging.error(f"Failed to send Teams notification: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Failed to send Teams notification: {e}")

def with_comprehensive_logging(pipeline_name: str, tech_center: Optional[str] = None):
    """Comprehensive logging decorator with error handling and notifications"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize logger and notifier
            logger = PipelineLogger(pipeline_name, tech_center)
            
            # Get config for notifications (if available)
            config = None
            if args and hasattr(args[0], 'config'):
                config = args[0].config
            elif 'config' in kwargs:
                config = kwargs['config']
            
            notifier = ErrorNotifier(config) if config else None
            
            try:
                # Log function start
                logger.log_stage_start(func.__name__, {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                
                # Execute original function
                result = func(*args, **kwargs)
                
                # Log successful completion
                metrics = {}
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
                
                logger.log_stage_complete(func.__name__, metrics)
                
                return result
                
            except Exception as e:
                # Log detailed error
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                
                error_details = logger.log_error(func.__name__, e, context)
                
                # Send notifications
                if notifier:
                    notifier.send_error_notification(error_details)
                else:
                    # Fallback console notification
                    print(f"🚨 ERROR in {func.__name__}: {str(e)}")
                
                # Re-raise the error
                raise
        
        return wrapper
    return decorator

# Simple version for basic error handling
def catch_errors(func):
    """Simple decorator to catch errors and log them"""
    
    def safe_function(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            print(f"✅ {func.__name__} completed successfully")
            logging.info(f"Function {func.__name__} completed successfully")
            return result
            
        except Exception as error:
            error_message = f"❌ ERROR in {func.__name__}: {error}"
            print(error_message)
            logging.error(f"Function {func.__name__} failed: {error}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Re-raise the error
            raise error
    
    return safe_function