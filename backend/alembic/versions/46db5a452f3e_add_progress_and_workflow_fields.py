"""Add progress to workflows and status to tasks (safe)

Revision ID: 46db5a452f3e
Revises: 9052b38a5136
Create Date: 2025-08-24 22:18:26.776435
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "46db5a452f3e"
down_revision: Union[str, None] = "9052b38a5136"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ✅ Check if column exists before adding
    conn = op.get_bind()

    # Add progress to workflows if missing
    if not column_exists(conn, "workflows", "progress"):
        op.add_column("workflows", sa.Column("progress", sa.String(), nullable=True))

    # Add status to tasks if missing
    if not column_exists(conn, "tasks", "status"):
        op.add_column("tasks", sa.Column("status", sa.String(), nullable=True))


def downgrade() -> None:
    # ✅ Safe drop (only if column exists)
    conn = op.get_bind()

    if column_exists(conn, "tasks", "status"):
        op.drop_column("tasks", "status")

    if column_exists(conn, "workflows", "progress"):
        op.drop_column("workflows", "progress")


# Utility: check if column exists
def column_exists(conn, table_name: str, column_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name=:t AND column_name=:c"
        ),
        {"t": table_name, "c": column_name},
    )
    return result.first() is not None
