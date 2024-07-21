GCP_PROJECT=test-setup-405206
IMAGE_NAME=insurance-churn
VERSION=1.0

build:
	docker build  -t "${IMAGE_NAME}" .

run:
	docker run -p 8000:8000 "${IMAGE_NAME}"

docker-auth:
	gcloud auth configure-docker

tag:
	docker tag "${IMAGE_NAME}" "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:${VERSION}"

push:
	docker push "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:${VERSION}"

cloud-build:
	gcloud builds submit --tag "gcr.io/${GCP_PROJECT}/${IMAGE_NAME}:${VERSION}"