"""create tables

Revision ID: 7db5aa33e0d4
Revises: bf65c2ad0097
Create Date: 2024-10-08 14:13:51.045112

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7db5aa33e0d4'
down_revision: Union[str, None] = 'bf65c2ad0097'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
