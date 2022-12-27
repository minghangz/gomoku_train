import os

if __name__=='__main__':
    for year in range(2009, 2023):
        url = f'https://gomocup.org/static/tournaments/{year}/results/gomocup{year}results.zip'
        os.system(f'wget {url} ./')
        os.system(f'unzip gomocup{year}results.zip -d gomocup{year}results/')
        os.system(f'rm gomocup{year}results.zip')
    os.system('mv gomocup2013results/gomocup2013arch/* gomocup2013results/')
