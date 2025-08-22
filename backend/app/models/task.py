from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, ForeignKey, Float  # noqa : E501
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.connection import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)

    # Task definition
    node_id = Column(String(100), nullable=False)  # Unique within workflow
    task_type = Column(String(100), nullable=False)  # csv_loader, preprocess, train_logreg, etc.  # noqa : E501
    task_name = Column(String(255), nullable=False)

    # Task configuration
    parameters = Column(JSON)  # Input parameters for the task
    input_data = Column(JSON)  # References to input datasets/models
    output_data = Column(JSON)  # References to output datasets/models

    # Execution info
    status = Column(String(50), default="pending")  # pending, running, completed, failed  # noqa : E501
    result = Column(JSON)  # Task execution results
    error_message = Column(Text)
    execution_time = Column(Float)  # seconds

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    workflow = relationship("Workflow", back_populates="tasks")
