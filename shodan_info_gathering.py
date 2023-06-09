#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
  Esse arquivo faz 
  verificação de 
  vulnerabilidade e a
  captura de banners de sites 
  alvos.  


  Modificado em 23 de dezembro de 2016
  por Vitor Mazuco (vitor.mazuco@gmail.com)
"""

import sys
import requests

# Importação das bibliotecas necessárias nesta aula
try:
    import shodan
except:
    sys.exit("[!] Por favor, intale as bibliotecas shodan e simplejson com os comandos: pip2 install simplejson e easy_install shodan")

#SHODAN_API_KEY = "b8sVHKAoo7gtPIP6G1Z6qa9tApRYd1CN" # Coloque o seu KEY API nessa parte!
SHODAN_API_KEY = "UZTIL9ochgBZCLGE75xr2EJWxJMeE1wf"
api = shodan.Shodan(SHODAN_API_KEY)

target = raw_input("https://www.facebook.com")

dnsResolve = 'https://api.shodan.io/dns/resolve?hostnames=' + target + '&key=' + SHODAN_API_KEY


try:
    # Primeiro, precisamos resolver nosso domínio de destinos para um IP
    resolved = requests.get(dnsResolve)
    hostIP = resolved.json()[target]
   

    # Então nós precisamos fazer uma pesquisa Shodan nesse IP
    host = api.host(hostIP)
    print %s % host['ip_str']
    print  %s % host.get('org', 'n/a')
    print  %s % host.get('os', 'n/a')


    # Imprimir todos os banners
    for item in host['data']:
        print %s % item['port']
        print %s % item['data']
        

    # Imprimir informações de vuln
    for item in host['vulns']:
        CVE = item.replace('!','')
        print  %s % item  # Printa as vulnerabilidade encontradas
        exploits = api.exploits.search(CVE)
        for item in exploits['matches']:
            if item.get('cve')[0] == CVE:
                print (item.get('Descrição'))
except:
    'Ocorreu um erro'
