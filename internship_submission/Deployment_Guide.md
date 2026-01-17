# 🚀 Employee Attrition Predictor - Deployment Guide

## Deployment Options

### 1. 🌐 **Streamlit Cloud (Easiest - Free)**

**Perfect for internship demos and portfolios!**

#### Steps:
1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Employee Attrition Predictor"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `app.py`
   - Deploy!

3. **Access your app:**
   - Get a public URL like: `https://nihalsmk-employee-attrition-predictor-app-6kizao.streamlit.app`
   - Share with recruiters and interviewers!

#### Requirements:
- GitHub account (free)
- Public repository
- `requirements_deploy.txt` file (✅ already created)

---

### 2. 🐳 **Docker Deployment (Professional)**

**Great for enterprise environments and cloud platforms!**

#### Local Docker:
```bash
# Build the image
docker build -t employee-attrition-predictor .

# Run the container
docker run -p 8501:8501 employee-attrition-predictor

# Access at: http://localhost:8501
```

#### Docker Compose (with database):
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
```

---

### 3. ☁️ **Cloud Platform Deployment**

#### **AWS (Amazon Web Services):**
```bash
# Using AWS App Runner
aws apprunner create-service \
  --service-name employee-attrition-predictor \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "your-ecr-repo/employee-attrition-predictor",
      "ImageConfiguration": {
        "Port": "8501"
      }
    }
  }'
```

#### **Google Cloud Platform:**
```bash
# Deploy to Cloud Run
gcloud run deploy employee-attrition-predictor \
  --image gcr.io/your-project/employee-attrition-predictor \
  --platform managed \
  --port 8501 \
  --allow-unauthenticated
```

#### **Microsoft Azure:**
```bash
# Deploy to Container Instances
az container create \
  --resource-group myResourceGroup \
  --name employee-attrition-predictor \
  --image your-registry/employee-attrition-predictor \
  --ports 8501 \
  --dns-name-label employee-attrition-app
```

---

### 4. 🖥️ **Local Development Server**

**For development and testing:**

```bash
# Install dependencies
pip install -r requirements_deploy.txt

# Run the training pipeline (first time only)
python main.py

# Start the web application
streamlit run app.py

# Access at: http://localhost:8501
```

---

### 5. 🏢 **Enterprise API Deployment**

Create a REST API for enterprise integration:

```python
# api.py - FastAPI version
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Employee Attrition Predictor API")

class EmployeeData(BaseModel):
    age: int
    department: str
    monthly_income: float
    # ... other fields

@app.post("/predict")
async def predict_attrition(employee: EmployeeData):
    # Load model and make prediction
    # Return risk score and recommendations
    pass
```

Deploy with:
```bash
# Using Uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000

# Using Docker
docker run -p 8000:8000 employee-attrition-api
```

---

## 📋 **Pre-Deployment Checklist**

### ✅ **Required Files:**
- [x] `app.py` - Streamlit web application
- [x] `requirements_deploy.txt` - Python dependencies
- [x] `Dockerfile` - Container configuration
- [x] `models/` - Trained ML models
- [x] `data/` - Sample dataset
- [x] `src/` - Source code modules

### ✅ **Before Deploying:**
1. **Run training pipeline:**
   ```bash
   python main.py
   ```

2. **Test locally:**
   ```bash
   streamlit run app.py
   ```

3. **Verify all files exist:**
   - `models/best_model.pkl`
   - `models/feature_encoders.pkl`
   - `data/hr_employee_data.csv`

4. **Check requirements:**
   ```bash
   pip install -r requirements_deploy.txt
   ```

---

## 🎯 **Recommended for Internships**

### **Option 1: Streamlit Cloud** ⭐
- **Pros:** Free, easy, public URL, great for demos
- **Cons:** Limited resources, public only
- **Best for:** Portfolio, interviews, presentations

### **Option 2: Local + Docker** ⭐⭐
- **Pros:** Professional setup, works offline, full control
- **Cons:** Requires Docker knowledge
- **Best for:** Technical interviews, on-site demos

### **Option 3: Cloud Platform** ⭐⭐⭐
- **Pros:** Enterprise-grade, scalable, impressive
- **Cons:** May cost money, more complex
- **Best for:** Advanced portfolios, real deployments

---

## 🔧 **Troubleshooting**

### Common Issues:

1. **Missing models:**
   ```bash
   python main.py  # Re-run training
   ```

2. **Import errors:**
   ```bash
   pip install -r requirements_deploy.txt
   ```

3. **Port conflicts:**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **Memory issues:**
   - Use smaller dataset
   - Reduce model complexity
   - Add swap space

---

## 📞 **Support**

For deployment help:
1. Check the logs: `streamlit run app.py --logger.level debug`
2. Verify file paths and permissions
3. Test with minimal dataset first
4. Use Docker for consistent environments

**Your Employee Attrition Predictor is ready for professional deployment! 🚀**