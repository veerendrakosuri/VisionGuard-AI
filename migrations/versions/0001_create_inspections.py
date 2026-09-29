"""Create inspections table."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "inspections",
        sa.Column("inspection_id", sa.String(length=36), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("decision", sa.String(length=4), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("threshold_source", sa.String(length=32), nullable=False),
        sa.Column("inference_time_ms", sa.Float(), nullable=False),
        sa.Column("heatmap_url", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("backbone", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("inspection_id"),
    )
    op.create_index("ix_inspections_decision", "inspections", ["decision"])
    op.create_index("ix_inspections_category", "inspections", ["category"])
    op.create_index("ix_inspections_created_at", "inspections", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_inspections_created_at", table_name="inspections")
    op.drop_index("ix_inspections_category", table_name="inspections")
    op.drop_index("ix_inspections_decision", table_name="inspections")
    op.drop_table("inspections")
