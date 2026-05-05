from pydantic import BaseModel
from typing import Optional

class TrainingParams(BaseModel):
    input_chunk_len: Optional[int] = None
    output_chunk_len: Optional[int] = None
    n_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    hidden_size: Optional[int] = None
    ff_size: Optional[int] = None
    num_blocks: Optional[int] = None
    dropout: Optional[float] = None
    learning_rate: Optional[float] = None
