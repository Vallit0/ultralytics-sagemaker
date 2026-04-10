# 🚀 AWS Machine Learning — Demo Completa

## Arquitectura

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  Kaggle   │───▶│  Local   │───▶│  S3 Bucket   │───▶│  SageMaker   │───▶│ Endpoint │
│  Dataset  │    │  (CSV)   │    │  (raw data)  │    │  Training Job │    │ (deploy) │
└──────────┘    └──────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

**Servicios:** S3, SageMaker, IAM  
**Dataset:** [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) (Kaggle)  
**Algoritmo:** XGBoost (clasificación binaria: sobrevive o no)

---

# PARTE 1 — AWS CLI

---

## Pre-requisitos

```bash
# Instalar AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Instalar Kaggle CLI
pip install kaggle

# Configurar AWS
aws configure
# AWS Access Key ID: ********
# AWS Secret Access Key: ********
# Default region: us-east-1
# Default output format: json

# Configurar Kaggle (necesitas tu API token de https://www.kaggle.com/settings)
mkdir -p ~/.kaggle
# cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## Paso 1 — Descargar dataset de Kaggle

```bash
# Descargar el dataset Titanic
kaggle competitions download -c titanic -p ./titanic-data
unzip ./titanic-data/titanic.zip -d ./titanic-data

ls ./titanic-data/
# train.csv  test.csv  gender_submission.csv
```

## Paso 2 — Preparar datos para SageMaker XGBoost

SageMaker XGBoost requiere que la **columna target sea la primera** y que **no haya headers**. Eliminamos columnas no numéricas para simplificar.

```bash
pip install pandas

cat > preprocess.py << 'PYEOF'
import pandas as pd

df = pd.read_csv("./titanic-data/train.csv")

# Seleccionar features numéricas relevantes
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()

# Encode Sex: male=1, female=0
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

# Rellenar Age nulos con la mediana
df["Age"] = df["Age"].fillna(df["Age"].median())

# Split 80/20
train = df.sample(frac=0.8, random_state=42)
validation = df.drop(train.index)

# Guardar SIN headers (requerimiento de XGBoost en SageMaker)
train.to_csv("./titanic-data/train_processed.csv", index=False, header=False)
validation.to_csv("./titanic-data/validation_processed.csv", index=False, header=False)

print(f"Train: {len(train)} filas | Validation: {len(validation)} filas")
print(f"Columnas: Survived, Pclass, Sex, Age, SibSp, Parch, Fare")
PYEOF

python preprocess.py
```

## Paso 3 — Crear bucket S3 y subir datos

```bash
BUCKET_NAME="ml-titanic-demo-$(date +%s)"
REGION="us-east-1"

aws s3 mb s3://$BUCKET_NAME --region $REGION

aws s3 cp ./titanic-data/train_processed.csv \
  s3://$BUCKET_NAME/data/train/train.csv

aws s3 cp ./titanic-data/validation_processed.csv \
  s3://$BUCKET_NAME/data/validation/validation.csv

# Verificar
aws s3 ls s3://$BUCKET_NAME/data/ --recursive
```

## Paso 4 — Crear IAM Role para SageMaker

```bash
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
  --role-name SageMakerTitanicRole \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --role-name SageMakerTitanicRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name SageMakerTitanicRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

ROLE_ARN=$(aws iam get-role --role-name SageMakerTitanicRole \
  --query 'Role.Arn' --output text)

echo "Role ARN: $ROLE_ARN"
sleep 10
```

## Paso 5 — Lanzar Training Job

```bash
IMAGE_URI="683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest"
TRAINING_JOB_NAME="titanic-xgb-$(date +%s)"

aws sagemaker create-training-job \
  --training-job-name $TRAINING_JOB_NAME \
  --role-arn $ROLE_ARN \
  --algorithm-specification \
      TrainingImage=$IMAGE_URI,TrainingInputMode=File \
  --resource-config \
      InstanceType=ml.m5.large,InstanceCount=1,VolumeSizeInGB=5 \
  --input-data-config '[
    {
      "ChannelName": "train",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://'"$BUCKET_NAME"'/data/train/",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "text/csv"
    },
    {
      "ChannelName": "validation",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "s3://'"$BUCKET_NAME"'/data/validation/",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "text/csv"
    }
  ]' \
  --output-data-config \
      S3OutputPath=s3://$BUCKET_NAME/output/ \
  --stopping-condition MaxRuntimeInSeconds=600 \
  --hyper-parameters \
      num_round=100,objective=binary:logistic,eval_metric=auc,max_depth=5,eta=0.2

echo "Training job: $TRAINING_JOB_NAME"
```

## Paso 6 — Monitorear entrenamiento

```bash
aws sagemaker describe-training-job \
  --training-job-name $TRAINING_JOB_NAME \
  --query '[TrainingJobStatus, SecondaryStatus]' --output text

aws sagemaker wait training-job-completed-or-stopped \
  --training-job-name $TRAINING_JOB_NAME

aws sagemaker describe-training-job \
  --training-job-name $TRAINING_JOB_NAME \
  --query 'FinalMetricDataList' --output table

echo "✅ Training completado"
```

## Paso 7 — Crear modelo y desplegar endpoint

```bash
MODEL_NAME="titanic-model-$(date +%s)"

MODEL_ARTIFACT=$(aws sagemaker describe-training-job \
  --training-job-name $TRAINING_JOB_NAME \
  --query 'ModelArtifacts.S3ModelArtifacts' --output text)

aws sagemaker create-model \
  --model-name $MODEL_NAME \
  --primary-container \
      Image=$IMAGE_URI,ModelDataUrl=$MODEL_ARTIFACT \
  --execution-role-arn $ROLE_ARN

ENDPOINT_CONFIG="titanic-epc-$(date +%s)"

aws sagemaker create-endpoint-config \
  --endpoint-config-name $ENDPOINT_CONFIG \
  --production-variants \
      VariantName=AllTraffic,ModelName=$MODEL_NAME,InstanceType=ml.t2.medium,InitialInstanceCount=1

ENDPOINT_NAME="titanic-ep-$(date +%s)"

aws sagemaker create-endpoint \
  --endpoint-name $ENDPOINT_NAME \
  --endpoint-config-name $ENDPOINT_CONFIG

echo "⏳ Creando endpoint: $ENDPOINT_NAME"
aws sagemaker wait endpoint-in-service --endpoint-name $ENDPOINT_NAME
echo "✅ Endpoint listo"
```

## Paso 8 — Invocar (predicción)

```bash
# Hombre, 3ra clase, 25 años, solo, tarifa $7.25
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name $ENDPOINT_NAME \
  --content-type "text/csv" \
  --body "3,1,25,0,0,7.25" \
  prediction.json
cat prediction.json
# ~0.08 → baja probabilidad de sobrevivir

echo "---"

# Mujer, 1ra clase, 30 años, sola, tarifa $100
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name $ENDPOINT_NAME \
  --content-type "text/csv" \
  --body "1,0,30,0,0,100.0" \
  prediction2.json
cat prediction2.json
# ~0.95 → alta probabilidad de sobrevivir
```

---

## 🧹 Limpieza (CLI)

```bash
aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME
aws sagemaker delete-endpoint-config --endpoint-config-name $ENDPOINT_CONFIG
aws sagemaker delete-model --model-name $MODEL_NAME

aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

aws iam detach-role-policy --role-name SageMakerTitanicRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name SageMakerTitanicRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name SageMakerTitanicRole

rm -rf ./titanic-data trust-policy.json preprocess.py prediction*.json

echo "🧹 Todo limpio"
```

---

## 💰 Costos estimados

| Servicio | Costo aprox. |
|----------|-------------|
| S3 (almacenamiento + requests) | ~$0.00 |
| SageMaker Training (ml.m5.large, ~5 min) | ~$0.01 |
| SageMaker Endpoint (ml.t2.medium, 30 min) | ~$0.03 |
| **Total demo** | **< $0.05** |

> ⚠️ **Eliminar el endpoint inmediatamente después de la demo.**

---
---

# PARTE 2 — Desde la Consola Web (UI)

Los mismos pasos pero desde el navegador.

---

## UI Paso 1 — Descargar dataset de Kaggle

1. Ir a [kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
2. Click en la pestaña **"Data"**
3. Click en **"Download All"** (necesitas cuenta de Kaggle)
4. Descomprimir el ZIP → `train.csv`, `test.csv`, `gender_submission.csv`
5. Ejecutar el script `preprocess.py` de la Parte 1 localmente para generar `train_processed.csv` y `validation_processed.csv`

> 💡 El preprocesamiento es local porque SageMaker XGBoost necesita un formato específico (target en primera columna, sin headers).

---

## UI Paso 2 — Crear bucket S3 y subir datos

1. Ir a **[console.aws.amazon.com/s3](https://console.aws.amazon.com/s3/)**
2. Click **"Create bucket"**
3. Configurar:
   - **Bucket name:** `ml-titanic-demo-{tu-nombre}`
   - **Region:** `US East (N. Virginia) us-east-1`
   - Dejar todo lo demás por defecto
4. Click **"Create bucket"**
5. Entrar al bucket → **"Create folder"** → `data`
6. Dentro de `data/` crear dos carpetas: `train/` y `validation/`
7. Entrar a `data/train/` → **"Upload"** → Subir `train_processed.csv`
8. Entrar a `data/validation/` → **"Upload"** → Subir `validation_processed.csv`
9. Crear otra carpeta en la raíz: `output/`

**Estructura final:**
```
ml-titanic-demo-{nombre}/
├── data/
│   ├── train/
│   │   └── train_processed.csv
│   └── validation/
│       └── validation_processed.csv
└── output/
```

---

## UI Paso 3 — Crear IAM Role

1. Ir a **[console.aws.amazon.com/iam](https://console.aws.amazon.com/iam/)**
2. Menú izquierdo → **Roles** → **"Create role"**
3. **Trusted entity type:** AWS service
4. **Use case:** Buscar y seleccionar **SageMaker**
5. Click **Next**
6. Verificar que `AmazonSageMakerFullAccess` está adjunto
7. Buscar y agregar también `AmazonS3FullAccess`
8. Click **Next**
9. **Role name:** `SageMakerTitanicRole`
10. Click **"Create role"**
11. Entrar al rol y **copiar el ARN** → Ej: `arn:aws:iam::123456789012:role/SageMakerTitanicRole`

---

## UI Paso 4 — Crear Training Job en SageMaker

1. Ir a **[console.aws.amazon.com/sagemaker](https://console.aws.amazon.com/sagemaker/)**
2. Menú izquierdo → **Training** → **Training jobs** → **"Create training job"**

**Job settings:**

| Campo | Valor |
|-------|-------|
| Job name | `titanic-xgb-demo` |
| IAM role | Seleccionar `SageMakerTitanicRole` |

**Algorithm:**

| Campo | Valor |
|-------|-------|
| Algorithm source | **Amazon SageMaker built-in algorithm** |
| Algorithm | **XGBoost** |

**Resource configuration:**

| Campo | Valor |
|-------|-------|
| Instance type | `ml.m5.large` |
| Instance count | `1` |
| Volume size | `5 GB` |

**Hyperparameters** (click "Add hyperparameter" para cada uno):

| Key | Value |
|-----|-------|
| `num_round` | `100` |
| `objective` | `binary:logistic` |
| `eval_metric` | `auc` |
| `max_depth` | `5` |
| `eta` | `0.2` |

**Input data configuration:**

*Channel 1 — train:*

| Campo | Valor |
|-------|-------|
| Channel name | `train` |
| S3 location | `s3://ml-titanic-demo-{nombre}/data/train/` |
| Content type | `text/csv` |

*Channel 2 (click "Add channel") — validation:*

| Campo | Valor |
|-------|-------|
| Channel name | `validation` |
| S3 location | `s3://ml-titanic-demo-{nombre}/data/validation/` |
| Content type | `text/csv` |

**Output data configuration:**

| Campo | Valor |
|-------|-------|
| S3 output path | `s3://ml-titanic-demo-{nombre}/output/` |

**Stopping condition:**

| Campo | Valor |
|-------|-------|
| Maximum runtime | `600` seconds |

3. Click **"Create training job"**
4. Esperar a que el status cambie a **Completed** (~3-5 min)

---

## UI Paso 5 — Crear Modelo

1. SageMaker → **Inference** → **Models** → **"Create model"**

| Campo | Valor |
|-------|-------|
| Model name | `titanic-model` |
| IAM role | `SageMakerTitanicRole` |

2. En **Container definition:**
   - **Inference code image:** `683313688378.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest`
   - **Model artifacts:** Copiar la ruta S3 desde el training job completado
     - Training jobs → `titanic-xgb-demo` → Output → Model artifact S3 URI
     - Ej: `s3://ml-titanic-demo-{nombre}/output/titanic-xgb-demo/output/model.tar.gz`

3. Click **"Create model"**

---

## UI Paso 6 — Crear Endpoint

**6a. Endpoint Configuration:**

1. SageMaker → **Inference** → **Endpoint configurations** → **"Create endpoint configuration"**

| Campo | Valor |
|-------|-------|
| Name | `titanic-epc` |

2. Click **"Add model"** → Seleccionar `titanic-model`

| Campo | Valor |
|-------|-------|
| Instance type | `ml.t2.medium` |
| Instance count | `1` |

3. Click **"Create endpoint configuration"**

**6b. Endpoint:**

1. SageMaker → **Inference** → **Endpoints** → **"Create endpoint"**

| Campo | Valor |
|-------|-------|
| Endpoint name | `titanic-endpoint` |
| Endpoint configuration | Seleccionar `titanic-epc` |

2. Click **"Create endpoint"**
3. Esperar status **InService** (~5-8 min)

---

## UI Paso 7 — Probar el Endpoint

No hay botón directo en la consola. Usar **CloudShell** (ícono de terminal en la barra superior de AWS):

```bash
# Hombre, 3ra clase, 25 años, solo, tarifa $7.25
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name titanic-endpoint \
  --content-type "text/csv" \
  --body "3,1,25,0,0,7.25" \
  /tmp/result.json && cat /tmp/result.json

# Mujer, 1ra clase, 30 años, sola, tarifa $100
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name titanic-endpoint \
  --content-type "text/csv" \
  --body "1,0,30,0,0,100.0" \
  /tmp/result2.json && cat /tmp/result2.json
```

**Interpretación:** Cercano a 0 → no sobrevive | Cercano a 1 → sobrevive

---

## UI Paso 8 — Limpieza desde la Consola

> ⚠️ **Hacer esto INMEDIATAMENTE después de la demo.**

1. **Endpoint:** SageMaker → Inference → Endpoints → `titanic-endpoint` → Actions → **Delete**
2. **Endpoint Config:** SageMaker → Inference → Endpoint configurations → `titanic-epc` → Actions → **Delete**
3. **Modelo:** SageMaker → Inference → Models → `titanic-model` → Actions → **Delete**
4. **Bucket S3:** S3 → Seleccionar bucket → **"Empty"** → luego **"Delete"**
5. **IAM Role:** IAM → Roles → `SageMakerTitanicRole` → **Delete**

---

## 📋 Tabla resumen CLI vs UI

| Paso | CLI | Consola UI |
|------|-----|-----------|
| Dataset | `kaggle competitions download` | Descargar ZIP desde kaggle.com |
| Preprocesar | Script Python local | Script Python local |
| Bucket + upload | `aws s3 mb` + `aws s3 cp` | S3 Console → Create + Upload |
| IAM Role | `aws iam create-role` | IAM Console → Create role |
| Training | `aws sagemaker create-training-job` | SageMaker → Create training job |
| Modelo | `aws sagemaker create-model` | SageMaker → Create model |
| Endpoint | `aws sagemaker create-endpoint` | SageMaker → Create endpoint |
| Inferencia | `invoke-endpoint` | CloudShell |
| Limpieza | `delete-*` + `s3 rb` | Delete desde cada consola |
