import sagemaker

session = sagemaker.Session()
bucket = "ivadas-data-722396408893"

patchcore_s3_path = session.upload_data(
    path="patchcore_model.tar.gz",
    bucket=bucket,
    key_prefix="models/patchcore-mvtec-multiclass",
)

print("Uploaded model artifact to:", patchcore_s3_path)
