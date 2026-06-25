"""
store.py — Persistência dos dados brutos em disco.

Responsabilidade única: ler e escrever DataFrames em data/raw/.
Não fala com a API (isso é trabalho do collect.py) — só com o disco.

Usamos Parquet em vez de CSV porque ele preserva os tipos das colunas
(int, float) sem reparsing, é compacto e é o formato padrão de pipeline
de dados. CSV viraria texto e perderia a tipagem na releitura.
"""

from pathlib import Path

import pandas as pd

# Pasta dos dados brutos. É relativa à raiz do projeto (um nível acima de src/).
# Fica fora do git (gitignored) porque é recriável a partir da API.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _season_path(year: int) -> Path:
    """Caminho do arquivo Parquet de uma temporada. Ex: .../season_2024.parquet"""
    return RAW_DIR / f"season_{year}.parquet"


def save_season(df: pd.DataFrame, year: int) -> Path:
    """
    Salva os resultados de uma temporada em data/raw/season_{year}.parquet.

    Args:
        df: DataFrame com os resultados (saída de get_season_results).
        year: ano da temporada, usado no nome do arquivo.

    Returns:
        O caminho do arquivo gravado.
    """
    # Falha explícita: não faz sentido salvar um DataFrame vazio
    if df.empty:
        raise ValueError(f"DataFrame vazio — nada para salvar na temporada {year}")

    # Cria data/raw/ se ainda não existir (parents=True cria data/ também)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = _season_path(year)
    df.to_parquet(path, index=False)
    return path


def listar_temporadas() -> list[int]:
    """
    Descobre quais temporadas estão salvas em data/raw/, lendo o disco.

    Fonte única de verdade dos anos disponíveis: em vez de chumbar uma lista
    de anos em cada módulo (features, train, predict, app), todos perguntam
    aqui. Assim, coletar um ano novo (ex: 2025) já entra no pipeline sozinho,
    sem editar código — coerente com a filosofia de pipeline reprodutível.

    Returns:
        Lista de anos em ordem crescente (ex: [2017, 2018, ..., 2024]).
    """
    # Varre os arquivos season_*.parquet e extrai o ano do nome do arquivo.
    # path.stem -> "season_2024"; split("_")[1] -> "2024" -> int.
    anos = [int(path.stem.split("_")[1]) for path in RAW_DIR.glob("season_*.parquet")]

    # Falha explícita: sem nenhuma temporada no disco não há o que processar.
    if not anos:
        raise FileNotFoundError(
            f"Nenhuma temporada em {RAW_DIR}. Rode a coleta primeiro: "
            "python src/collect_seasons.py"
        )

    # Ordena crescente — a ordem cronológica importa p/ a validação temporal.
    return sorted(anos)


def load_season(year: int) -> pd.DataFrame:
    """
    Lê os resultados de uma temporada salva em disco.

    Permite que os próximos módulos (features, train) leiam os dados sem
    rebater na API a cada execução.

    Args:
        year: ano da temporada a carregar.

    Returns:
        DataFrame com os resultados gravados.
    """
    path = _season_path(year)

    # Falha explícita e com mensagem útil se o arquivo não existir
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode a coleta primeiro (collect.get_season_results)."
        )

    return pd.read_parquet(path)


# Teste rápido: coleta 2024 via collect, salva e relê para confirmar o round-trip.
if __name__ == "__main__":
    from collect import get_season_results

    df = get_season_results(2024)
    caminho = save_season(df, 2024)
    print(f"\nSalvo em: {caminho}")

    # Relê e confere se voltou igual (mesmo número de linhas e tipos preservados)
    recarregado = load_season(2024)
    print(f"Recarregado: {len(recarregado)} linhas")
    print(f"Tipos preservados:\n{recarregado.dtypes}")
