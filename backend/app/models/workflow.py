from app.database.connection import Base
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,  # noqa :
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Workflow definition (nodes and connections)
    nodes = Column(JSON)  # List of workflow nodes
    connections = Column(JSON)  # Node connections/edges

    # Execution info
    status = Column(
        String(50), default="draft"
    )  # draft, pending, running, completed, failed
    progress = Column(Float, default=0.0)  # Progress percentage (0-100)
    results = Column(JSON)  # Final results
    execution_time = Column(Float)  # seconds

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))  # When execution started
    completed_at = Column(DateTime(timezone=True))  # When execution finished
    is_template = Column(Boolean, default=False)

    # Relationships
    tasks = relationship(
        "Task", back_populates="workflow", cascade="all, delete-orphan"
    )

    datasets = relationship(
        "Dataset", back_populates="workflow", cascade="all, delete-orphan"
    )
