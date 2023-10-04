FROM python:3-alpine

RUN apk update
RUN apk add make automake gcc g++

WORKDIR /usr/src/app

RUN pip install --upgrade pip setuptools wheel
RUN pip install pandas numpy scikit-learn matplotlib

COPY . .

CMD ["/bin/sh"]
