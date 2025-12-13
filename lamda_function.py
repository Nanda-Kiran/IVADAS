import os
import json
import base64
import hashlib

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
runtime = boto3.client("sagemaker-runtime")

# You can hard-code these or set them as Lambda environment variables
CACHE_BUCKET = os.environ.get("CACHE_BUCKET", "ivadas-predict-cache-722396408893")
PATCHCORE_EP = os.environ.get("PATCHCORE_ENDPOINT", "patchcore-mvtec-all-classes")
SOFTPATCH_EP = os.environ.get("SOFTPATCH_ENDPOINT", "softpatch-mvtec-all-classes")


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _parse_body(event):
    """Handle API Gateway proxy (HTTP/REST) event and return parsed JSON dict."""
    body_raw = event.get("body", "")
    if event.get("isBase64Encoded"):
        body_bytes = base64.b64decode(body_raw)
    else:
        body_bytes = body_raw.encode("utf-8")
    return json.loads(body_bytes.decode("utf-8"))


def lambda_handler(event, context):
    try:
        # 1) Parse incoming JSON
        body = _parse_body(event)

        # Expect client JSON: { "class_name", "image_base64", "model" }
        img_b64 = body["image_base64"]
        img_bytes = base64.b64decode(img_b64)

        class_name = body.get("class_name", "bottle")
        model_choice = body.get("model", "patchcore")  # "patchcore" or "softpatch"

        if model_choice not in ("patchcore", "softpatch"):
            model_choice = "patchcore"

        endpoint = PATCHCORE_EP if model_choice == "patchcore" else SOFTPATCH_EP

        # 2) Compute cache key (model + SHA256(image bytes))
        img_hash = sha256_bytes(img_bytes)
        cache_key = f"mvtec_cache/{model_choice}/{img_hash}.json"

        # 3) Try cache first
        try:
            obj = s3.get_object(Bucket=CACHE_BUCKET, Key=cache_key)
            cached_result = json.loads(obj["Body"].read())
            cached_result["from_cache"] = True

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(cached_result),
            }

        except ClientError as e:
            # If it's not a "key not found" error, bubble up
            if e.response["Error"]["Code"] != "NoSuchKey":
                raise
            # else: cache miss → keep going

        # 4) Cache miss → call the appropriate SageMaker endpoint
        if model_choice == "patchcore":
            # matches your working PatchCore test code
            sm_payload = {
                "class_name": class_name,
                "image_base64": img_b64,
            }
        else:
            # matches your working SoftPatch test code
            sm_payload = {
                "classname": class_name,
                "image": img_b64,
            }

        sm_response = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(sm_payload),
        )

        sm_body = json.loads(sm_response["Body"].read().decode("utf-8"))

        # Normalize response keys so the frontend always gets the same shape
        anomaly_score = sm_body.get("anomaly_score")
        returned_class = (
            sm_body.get("class_name")
            or sm_body.get("classname")
            or class_name
        )

        result = {
            "anomaly_score": anomaly_score,
            "class_name": returned_class,
            "model": model_choice,
            "from_cache": False,
        }

        # If SoftPatch returns a heatmap, pass it through transparently
        if "anomaly_heatmap" in sm_body:
            result["anomaly_heatmap"] = sm_body["anomaly_heatmap"]

        # 5) Write to cache
        s3.put_object(
            Bucket=CACHE_BUCKET,
            Key=cache_key,
            Body=json.dumps(result).encode("utf-8"),
            ContentType="application/json",
        )

        # 6) Return response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result),
        }

    except Exception as e:
        # Useful for debugging failures
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
