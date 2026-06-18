"""
collect_seasons.py — Coleta várias temporadas de uma vez e salva no disco.

Orquestra os módulos que já existem: usa collect.get_season_results para
buscar cada temporada na API e store.save_season para gravar em Parquet.
Não tem lógica de dados própria — é só o "maestro" que junta as peças.
"""

from collect import get_season_results  # busca uma temporada inteira na API
from store import save_season           # grava o DataFrame em data/raw/

# Anos que queremos no dataset. range(2021, 2025) gera 2021, 2022, 2023, 2024
# (o limite de cima é exclusivo). 4 temporadas dão base p/ validação temporal:
# treinar nas antigas, testar na mais recente.
ANOS = range(2021, 2025)


def coletar_temporadas(anos) -> None:
    """
    Coleta e salva cada temporada do intervalo informado.

    Args:
        anos: iterável de anos (ex: range(2021, 2025) ou [2023, 2024]).
    """
    # Percorre ano a ano — cada iteração é uma temporada independente
    for ano in anos:
        print(f"\n{'='*50}")           # separador visual entre temporadas
        print(f"Coletando temporada {ano}...")
        print('='*50)

        # Busca todas as corridas do ano (já respeita o rate limit lá dentro)
        df = get_season_results(ano)

        # Grava em data/raw/season_{ano}.parquet e guarda o caminho devolvido
        caminho = save_season(df, ano)

        # Confirma o que foi salvo: linhas totais e arquivo de destino
        print(f"-> {len(df)} linhas salvas em {caminho}")


# Roda a coleta completa quando executado direto: python src/collect_seasons.py
if __name__ == "__main__":
    coletar_temporadas(ANOS)  # dispara a coleta das 4 temporadas
    print("\nColeta de todas as temporadas concluída.")  # aviso final
