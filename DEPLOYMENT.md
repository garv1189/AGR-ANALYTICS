# 🚀 Deployment Guide - SEC Forensic Auditor

## Quick Start Options

### Option 1: Docker Compose (Recommended)

**Easiest way to deploy with all services:**

```bash
# Clone the repository
git clone https://github.com/garv1189/AGR-ANALYTICS.git
cd AGR-ANALYTICS

# Start all services with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

**Access:**
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Stop services:**
```bash
docker-compose down
```

### Option 2: Quick Start Script

```bash
./start.sh
```

### Option 3: Manual Setup

See main README.md for detailed manual setup instructions.

---

## Production Deployment

### Prerequisites

- Domain name configured
- SSL certificates (Let's Encrypt recommended)
- Cloud provider account (AWS, GCP, Azure, DigitalOcean)
- Docker & Docker Compose installed

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 2: Clone & Configure

```bash
# Clone repository
git clone https://github.com/garv1189/AGR-ANALYTICS.git
cd AGR-ANALYTICS

# Configure environment
cd backend
cp .env.example .env
nano .env  # Edit with production values

# Key settings to change:
# - DEBUG=False
# - SECRET_KEY=<generate-with-openssl-rand-hex-32>
# - POSTGRES_PASSWORD=<strong-password>
# - OPENAI_API_KEY=<your-key>
# - ANTHROPIC_API_KEY=<your-key>
```

### Step 3: SSL Configuration

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### Step 4: Deploy with Docker Compose

```bash
# Build and start services
docker-compose up -d --build

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### Step 5: Configure Nginx (Production)

Create `/etc/nginx/sites-available/forensic-auditor`:

```nginx
upstream backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Frontend
    location / {
        proxy_pass http://localhost:80;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
        
        # CORS
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS, DELETE, PUT';
        add_header Access-Control-Allow-Headers 'Content-Type, Authorization';
    }

    # WebSocket support (future)
    location /ws {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/forensic-auditor /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Cloud Platform Deployment

### AWS Deployment

#### Using EC2

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: t3.large or larger (for ML models)
   - Storage: 50GB+ SSD
   - Security Group: Allow ports 80, 443, 22

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Follow server setup steps above
   ```

3. **Configure Elastic IP** (optional but recommended)

#### Using ECS (Container Service)

```bash
# Install AWS CLI
pip install awscli

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URI

# Build and push images
docker build -t forensic-backend ./backend
docker tag forensic-backend:latest YOUR_ECR_URI/forensic-backend:latest
docker push YOUR_ECR_URI/forensic-backend:latest

# Create ECS task definition and service
# Use AWS Console or CloudFormation
```

### Google Cloud Platform (GCP)

#### Using Compute Engine

```bash
# Create VM
gcloud compute instances create forensic-auditor \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB

# SSH and setup
gcloud compute ssh forensic-auditor

# Follow server setup steps
```

#### Using Cloud Run

```bash
# Build and submit to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/forensic-backend ./backend

# Deploy to Cloud Run
gcloud run deploy forensic-backend \
    --image gcr.io/PROJECT_ID/forensic-backend \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### DigitalOcean

#### Using Droplets

1. Create Droplet (Ubuntu 22.04, 8GB RAM, 4 vCPUs)
2. Add SSH key
3. Connect and follow server setup steps
4. Configure firewall

#### Using App Platform

```yaml
# app.yaml
name: forensic-auditor
services:
  - name: backend
    dockerfile_path: backend/Dockerfile
    instance_count: 1
    instance_size_slug: professional-m
    routes:
      - path: /api
  - name: frontend
    dockerfile_path: frontend/Dockerfile
    instance_count: 1
    instance_size_slug: basic-xxs
    routes:
      - path: /
databases:
  - name: postgres
    engine: PG
    version: "15"
```

---

## Monitoring & Maintenance

### Health Checks

```bash
# Check backend health
curl http://your-domain.com:8000/health

# Check Docker services
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Monitoring Setup

#### Prometheus + Grafana

Add to `docker-compose.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
```

### Backup Strategy

```bash
# Backup databases
docker-compose exec postgres pg_dump -U forensic_user forensic_db > backup_$(date +%Y%m%d).sql
docker-compose exec mongo mongodump --out=/backup/mongo_$(date +%Y%m%d)

# Backup application data
tar -czf backup_$(date +%Y%m%d).tar.gz backend/uploads backend/models_cache
```

### Updates & Maintenance

```bash
# Pull latest code
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose up -d --build

# Clean up old images
docker system prune -a
```

---

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_filings_cik ON filings(cik);
CREATE INDEX idx_reports_risk_level ON forensic_reports(risk_level);
CREATE INDEX idx_companies_ticker ON companies(ticker);
```

### 2. Caching Strategy

Enable Redis caching in `.env`:
```
REDIS_HOST=redis
REDIS_PORT=6379
```

### 3. Model Caching

Pre-download ML models:
```bash
python -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('ProsusAI/finbert', cache_dir='./models_cache'); \
  AutoModel.from_pretrained('ProsusAI/finbert', cache_dir='./models_cache')"
```

### 4. Load Balancing

Use Nginx load balancing for multiple backend instances:

```nginx
upstream backend_cluster {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

---

## Security Hardening

### 1. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 2. Fail2Ban Setup

```bash
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 3. Docker Security

- Run containers as non-root user
- Use read-only file systems where possible
- Limit container resources
- Regular image updates

### 4. Environment Variables

Never commit `.env` files:
```bash
# Use secrets management
# AWS: AWS Secrets Manager
# GCP: Secret Manager
# Docker: Docker Secrets
```

---

## Troubleshooting

### Backend not starting

```bash
# Check logs
docker-compose logs backend

# Common issues:
# 1. Missing environment variables
# 2. Port already in use
# 3. Database connection failed
```

### Frontend not loading

```bash
# Check Nginx logs
docker-compose logs frontend

# Verify API URL in frontend .env
```

### ML models failing

```bash
# Ensure enough disk space
df -h

# Check memory
free -m

# Increase Docker memory limit if needed
```

### Database connection issues

```bash
# Test PostgreSQL connection
docker-compose exec postgres psql -U forensic_user -d forensic_db

# Reset database
docker-compose down -v
docker-compose up -d
```

---

## Scaling Considerations

### Horizontal Scaling

1. **Multiple Backend Instances**
   ```bash
   docker-compose up -d --scale backend=3
   ```

2. **Load Balancer**
   - Use Nginx, HAProxy, or cloud load balancer

3. **Database Replication**
   - PostgreSQL read replicas
   - MongoDB replica sets

### Vertical Scaling

- Increase server resources (CPU, RAM)
- Use GPU instances for ML inference
- SSD storage for databases

---

## Cost Optimization

### Cloud Provider Estimates

**AWS:**
- EC2 t3.xlarge: ~$150/month
- RDS PostgreSQL: ~$80/month
- S3 storage: ~$5/month
- **Total: ~$235/month**

**GCP:**
- Compute Engine n1-standard-4: ~$130/month
- Cloud SQL: ~$75/month
- Cloud Storage: ~$5/month
- **Total: ~$210/month**

**DigitalOcean:**
- Droplet (8GB, 4vCPUs): ~$48/month
- Managed PostgreSQL: ~$60/month
- Spaces: ~$5/month
- **Total: ~$113/month**

### Cost Reduction Tips

1. Use spot/preemptible instances
2. Auto-scaling based on demand
3. Scheduled shutdowns for non-production
4. Efficient model caching
5. CDN for static assets

---

## Support & Resources

- **Documentation**: See README.md
- **Issues**: GitHub Issues
- **Updates**: Check releases regularly
- **Security**: Report vulnerabilities responsibly

---

**🎉 Your SEC Forensic Auditor is now production-ready!**
