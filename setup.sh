mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"582613@stud.hvl.no\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml