import os
import boto3
from botocore.exceptions import ClientError

sender = os.environ.get("ALERT_EMAIL_FROM")
recipient = os.environ.get("ALERT_EMAIL_TO")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

if not sender or not recipient:
    raise ValueError("Missing ALERT_EMAIL_FROM or ALERT_EMAIL_TO environment variable.")

subject = "✅ SES Test Email"
body_text = f"""Hello,

This is a test email sent via Amazon SES using boto3.

From: {sender}
To:   {recipient}
Region: {region}

If you received this message, your SES setup is working.
"""

client = boto3.client("ses", region_name=region)

try:
    response = client.send_email(
        Source=sender,
        Destination={"ToAddresses": [recipient]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": body_text}},
        },
    )
    print("✅ Email sent! Message ID:", response["MessageId"])
except ClientError as e:
    print("❌ Failed to send email:")
    print(e.response["Error"]["Message"])
