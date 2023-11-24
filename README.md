#Запуск и работа с системой  

1. Отредактировать файлы `.env`
2. Отредактировать конфигурационные файлы в каталоге `configs`
3. Запустить команду `docker-compose up -d` дождаться запуска всех контейнеров
4. Ввести отредактировав под свои нужды команду:
docker exec -it data_science_ui superset fab create-admin \
			   --username admin \
			   --firstname Superset \
			   --lastname Admin \
			   --email admin@admin.com \
			   --password hf#d,mIDN5dhI*C539JF; \
docker exec -it data_science_ui superset db upgrade; \
docker exec -it data_science_ui superset init;

5. Ссылки для работы в системе:
 - ip_host:8089 - Superset (логин и пароль в пункте 4 текущего документа)
 - ip_host:3025/docs - API
 
6. Авторизоваться в Superset используя имя и пароль заданные выше
 - Загрузить Dashboard из сохраненного шаблона: 
 
7. Действующее Демо:  
https://board.vniizht.ru/superset/dashboard/p/n9VApnjQyjX/  
login: demo  
pass: 123456Qw@  

## Алгоритм
Notebook'и c алогоритмом находится в каталоге `research`