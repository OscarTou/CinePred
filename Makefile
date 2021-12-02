# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* CinePred/*.py

black:
	@black scripts/* CinePred/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr CinePred-*.dist-info
	@rm -fr CinePred.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#          Streamlit server
# ----------------------------------
streamlit:
	-@streamlit run CinePred/app.py

# ----------------------------------
#          Fast api
# ----------------------------------
run_api:
	uvicorn CinePred.api.fast:app --reload


# ----------------------------------
#         	Google Cloud
# ----------------------------------

# project id - replace with your GCP project id
PROJECT_ID='imposing-water-328017'

# bucket name - replace with your GCP bucket name
BUCKET_NAME= "wagon-data-722-cinepred"

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION='europe-west1'

set_project:
	gcloud config set project ${PROJECT_ID}

create_bucket:
	gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="/home/oscartouze/code/OscarTou/CinePred/raw_data/IMDb_movies.csv"
LOCAL_PATH_2="/home/oscartouze/code/OscarTou/CinePred/raw_data/cat_acteur.csv"
MODEL_PATH ="/home/oscartouze/code/OscarTou/CinePred/CinePred/models/model.joblib"
LOCAL_PATH_PREPRO = "/home/oscartouze/code/OscarTou/CinePred/raw_data/preprocessed.csv"
LOCAL_PATH_CURRENCIES="/home/oscartouze/code/OscarTou/CinePred/raw_data/currencies.csv"
LOCAL_PATH_IMAGES = "/home/oscartouze/code/OscarTou/CinePred/raw_data/images"
# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)

BUCKET_FOLDER=data
BUCKET_DATA_FOLDER=model


# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})
BUCKET_FILE_NAME_2=$(shell basename ${LOCAL_PATH_2})
BUCKET_FILE_NAME_PREPRO=$(shell basename ${LOCAL_PATH_PREPRO})
BUCKET_FILE_NAME_MODEL=$(shell basename ${MODEL_PATH})
BUCKET_FILE_NAME_CURRENCIES=$(shell basename ${LOCAL_PATH_CURRENCIES})
BUCKET_FILE_NAME_IMAGES=$(shell basename ${LOCAL_PATH_IMAGES})





upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

upload_data_2:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	gsutil cp ${LOCAL_PATH_2} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME_2}

upload_model:
	gsutil cp ${LOCAL_PATH_2} gs://${BUCKET_NAME}/${BUCKET_DATA_FOLDER}/${BUCKET_FILE_NAME_MODEL}

upload_preprocessed:
	gsutil cp ${LOCAL_PATH_PREPRO} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME_PREPRO}

upload_currencies:
	gsutil cp ${LOCAL_PATH_CURRENCIES} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME_CURRENCIES}

upload_images:
	gsutil cp -r ${LOCAL_PATH_IMAGES} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME_IMAGES}

# ----------------------------------
#         	Run locally
# ----------------------------------


PACKAGE_NAME=CinePred
FILENAME=new_model

run_locally:
	python -m ${PACKAGE_NAME}.${FILENAME}

# ----------------------------------
#          GCP_submit_training
# ----------------------------------

REGION=europe-west1

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15


JOB_NAME=cinepred_pipeline_$(shell date +'%Y%m%d_%H%M%S')


gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs

# DOCKER

build_docker :
	docker build --tag=eu.gcr.io/${PROJECT_ID}/cinepred .

docker_push :
	docker push eu.gcr.io/${PROJECT_ID}/cinepred

# create service à partir de l'image et hop, url -> url de request
run_docker_locally :
	docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/cinepred
