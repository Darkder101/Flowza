from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.connection import Base


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Workflow definition (nodes and connections)
    nodes = Column(JSON)  # List of workflow nodes
    connections = Column(JSON)  # Node connections/edges

    # Execution info
    status = Column(String(50), default="draft")  # draft, running, completed, failed  # noqa : E501
    results = Column(JSON)  # Final results
    execution_time = Column(Integer)  # seconds

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_template = Column(Boolean, default=False)

    # Relationships
    tasks = relationship("Task", back_populates="workflow", cascade="all, delete-orphan")  # noqa : E501
