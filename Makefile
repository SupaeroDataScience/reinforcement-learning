
all:
	mkdocs build

sync:
	cp docs/index.md README.md

deploy:
	git fetch
	git checkout master
	mkdocs gh-deploy

clean:
	rm -rf site
