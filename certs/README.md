# Certificate Storage Directory

This directory stores AWS IoT certificates and keys for secure MQTT communication.

## Files stored here:
- `AmazonRootCA1.pem` - Amazon Root CA certificate
- `device-certificate.pem.crt` - Device certificate 
- `device-private.pem.key` - Private key (keep secure!)
- `device-public.pem.key` - Public key

## Security Notes:
- **Never commit private keys to version control**
- Set appropriate file permissions (600 for private keys)
- Backup certificates securely
- Rotate certificates periodically as per security policy

## Setup:
Run the setup script to automatically download and configure certificates:
```bash
python setup_aws_iot.py --thing-name arxplorer-client
```