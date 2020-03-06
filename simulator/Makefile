run_app:
	docker build -t docker-libsumo .
	--docker stop libsumo_bench
	--docker rm libsumo_bench
	docker run --rm --name libsumo_bench -d docker-libsumo -v `pwd`:`pwd` -w `pwd` -i -t  ubuntu pwd

build:
	docker build -t docker-libsumo .

run: run_app
