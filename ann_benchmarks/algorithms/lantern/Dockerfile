FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone --recursive https://github.com/lanterndata/lantern.git /tmp/lantern


RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake postgresql postgresql-server-dev-all
RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/lantern && \
  mkdir build && \
  cd build && \
  cmake .. && \
	make install

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION lantern" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '4GB'" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '4GB'"
USER root

RUN pip install psycopg[binary] pgvector
