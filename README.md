# Lucas Script

Esse script, atualmente, procura todas as pastas começando na pasta do argumento e analiza todas as pastas que tiverem dentro dela

- Ignora as pastas os com status **ok** (o nome da pasta termina em **\*_ok**);

- Ignora as pastas que não tenham os arquivos basicos de um cálculo do *orca*;

- Analiza as pastas que tenha arquivos **\*.xyz**, **\*.imp**, e **slurm-\*.out**:
  - Copia a esturtura do arquivo **\*.xyz** para o arquivo **\*.imp**;

  - Lê os arquivos **slurm-\*.out**, e verifica se o último arquivo apresenta a mensagem de convergência

## Examples (and tests)

```bash
$ python3 the_script.py top_folder
```
