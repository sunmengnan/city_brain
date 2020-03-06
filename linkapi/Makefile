run_app:
	docker build -t docker-tmap .
	--docker stop tmap
	--docker rm tmap
	docker run -p 5001:5001 --name tmap -d docker-tmap
run: run_app