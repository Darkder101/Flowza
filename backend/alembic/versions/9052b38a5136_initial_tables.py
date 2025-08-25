"""Initial tables

Revision ID: 9052b38a5136
Revises:
Create Date: 2025-08-24 20:46:41.191814
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9052b38a5136"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop tables in correct order to respect foreign keys
    op.drop_table("tasks")
    op.drop_table("workflows")
    op.drop_table("datasets")


def downgrade() -> None:
    # Recreate workflows table first (parent)
    op.create_table(
        "workflows",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("name", sa.VARCHAR(length=255), nullable=False),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("nodes", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("connections", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.VARCHAR(length=50), nullable=True),
        sa.Column("results", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("execution_time", sa.INTEGER(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("is_template", sa.BOOLEAN(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="workflows_pkey"),
    )
    op.create_index("ix_workflows_id", "workflows", ["id"], unique=False)

    # Recreate tasks table (child)
    op.create_table(
        "tasks",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("workflow_id", sa.INTEGER(), nullable=False),
        sa.Column("node_id", sa.VARCHAR(length=100), nullable=False),
        sa.Column("task_type", sa.VARCHAR(length=100), nullable=False),
        sa.Column("task_name", sa.VARCHAR(length=255), nullable=False),
        sa.Column("parameters", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("input_data", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("output_data", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.VARCHAR(length=50), nullable=True),
        sa.Column("result", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.TEXT(), nullable=True),
        sa.Column("execution_time", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["workflow_id"], ["workflows.id"], name="tasks_workflow_id_fkey"
        ),
        sa.PrimaryKeyConstraint("id", name="tasks_pkey"),
    )
    op.create_index("ix_tasks_id", "tasks", ["id"], unique=False)

    # Recreate datasets table
    op.create_table(
        "datasets",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("name", sa.VARCHAR(length=255), nullable=False),
        sa.Column("file_path", sa.VARCHAR(length=500), nullable=False),
        sa.Column("format", sa.VARCHAR(length=50), nullable=True),
        sa.Column("size_mb", sa.DOUBLE_PRECISION(precision=53), nullable=True),
        sa.Column("num_rows", sa.INTEGER(), nullable=True),
        sa.Column("num_columns", sa.INTEGER(), nullable=True),
        sa.Column(
            "columns_info", postgresql.JSON(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("source", sa.VARCHAR(length=255), nullable=True),
        sa.Column("is_temporary", sa.BOOLEAN(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="datasets_pkey"),
    )
    op.create_index("ix_datasets_id", "datasets", ["id"], unique=False)
