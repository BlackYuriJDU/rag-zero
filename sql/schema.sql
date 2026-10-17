-- RAG Zero — esquema inicial (Postgres 15+ com pgvector)
-- Aplicar manualmente:  psql -f sql/schema.sql
-- Ou via docker-compose (montado em docker-entrypoint-initdb.d).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Contratações ingeridas da API pública do PNCP.
CREATE TABLE IF NOT EXISTS contratacoes (
    id                          BIGSERIAL PRIMARY KEY,
    numero_controle_pncp        TEXT UNIQUE NOT NULL,
    ano_compra                  INT,
    sequencial_compra           INT,
    objeto_compra               TEXT,
    valor_total_estimado        NUMERIC(16, 2),
    modalidade_nome             TEXT,
    situacao_compra_nome        TEXT,
    modo_disputa_nome           TEXT,
    srp                         BOOLEAN,
    processo                    TEXT,
    valor_total_homologado      NUMERIC(16, 2),
    uf                          CHAR(2),
    municipio                   TEXT,
    orgao_cnpj                  TEXT,
    orgao_razao_social          TEXT,
    data_publicacao             DATE,
    data_abertura_proposta      TIMESTAMPTZ,
    data_encerramento_proposta  TIMESTAMPTZ,
    link_sistema_origem         TEXT,
    raw                         JSONB NOT NULL,
    coletado_em                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_contratacoes_uf_data
    ON contratacoes (uf, data_publicacao DESC);
CREATE INDEX IF NOT EXISTS idx_contratacoes_modalidade
    ON contratacoes (modalidade_nome);

-- Documentos vinculados a uma contratação (editais, termos de referência, anexos).
-- Só o texto extraído vive aqui; os PDFs originais não são versionados.
CREATE TABLE IF NOT EXISTS documentos (
    id              BIGSERIAL PRIMARY KEY,
    contratacao_id  BIGINT REFERENCES contratacoes (id) ON DELETE CASCADE,
    tipo            TEXT,           -- edital | termo_de_referencia | anexo | ...
    url             TEXT,
    origem          TEXT,           -- pncp | externo
    sha256          TEXT UNIQUE,
    texto           TEXT,
    baixado_em      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunks prontos para recuperação híbrida (vetor + FTS português).
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    documento_id  BIGINT REFERENCES documentos (id) ON DELETE CASCADE,
    seq           INT NOT NULL,
    secao         TEXT,             -- seção estrutural do documento (ex.: "5.3")
    conteudo      TEXT NOT NULL,
    tokens        INT,
    -- 384 dims = e5-small multilingual. Ajustar caso o modelo de embedding mude.
    embedding     vector(384),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', conteudo)) STORED,
    UNIQUE (documento_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON chunks USING gin (tsv);
