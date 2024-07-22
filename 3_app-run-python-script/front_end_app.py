!pip uninstall transformers -y
!pip install transformers
!pip install streamlit
!streamlit run front_end2.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1
