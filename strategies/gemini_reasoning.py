import enum
from pydantic import BaseModel, Field
from google import genai
from config.settings import GEMINI_API_KEY

ai_client = genai.Client(api_key=GEMINI_API_KEY)

class TradeAction(str, enum.Enum):
    BUY = "BUY"
    HOLD = "HOLD"

class AdvancedTradingSignal(BaseModel):
    mathematical_proof: str = Field(description="Step-by-step mathematical proof evaluating the BOS and FVG rules using the exact prices from the JSON matrix.")
    action: TradeAction = Field(description="Must be BUY if ALL strategy confirmation criteria match perfectly, otherwise HOLD.")
    setup_type: str = Field(description="The underlying liquidity pool swept: 'PREV_DAY_LOW_SWEEP', 'FIRST_HOUR_LOW_SWEEP', or 'NONE'")
    bos_confirmed: bool = Field(description="True only if a candle BODY closed cleanly above/below the swing high/low in the provided matrix context. Wicks do not count.")
    fvg_entered: bool = Field(description="True if current market price has pulled back into a valid displacement Fair Value Gap zone identifiable anywhere within the provided 40-candle chart window.")
    calculated_stop_loss: float = Field(description="The exact initial stop loss price floor calculated for the leveraged execution asset.")
    calculated_tp1: float = Field(description="The target execution price for the leveraged asset when the anchor index hits its Today's Opening Price milestone.")
    calculated_tp2: float = Field(description="The final profit target price calculated for the leveraged execution asset placed just ahead of the opposing daily boundary.")
    reasoning: str = Field(description="A brief explanation of how market structure checks out across the broader 40-candle matrix context.")
    confidence_score: int = Field(description="Rate the structural cleanliness of the BOS and FVG on a scale of 1 to 100. 85+ requires textbook displacement and gap re-entry. 65-84 indicates a valid but slightly messy structure. Under 65 is invalid.")