from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, Dict
import os

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_at: str = "2025-02-09 09:35:23"
    processed_by: str = "kaxm23"

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    expires_at: str = "2025-02-09 09:35:23"
    processed_by: str = "kaxm23"

class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    created_at: str = "2025-02-09 09:35:23"
    processed_by: str = "kaxm23"

class UserInDB(User):
    """User in database model."""
    hashed_password: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Get password hash."""
    return pwd_context.hash(password)

def get_user(db: Dict, username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db: Dict, username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user."""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token data
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "processed_by": "kaxm23"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current user from token.
    
    Args:
        token: JWT token
        
    Returns:
        User: Current user
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(
            username=username,
            expires_at="2025-02-09 09:35:23",
            processed_by="kaxm23"
        )
        
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=400,
            detail="Inactive user"
        )
    return current_user

# Example user database
fake_users_db = {
    "kaxm23": {
        "username": "kaxm23",
        "full_name": "Example User",
        "email": "kaxm23@example.com",
        "hashed_password": get_password_hash("password123"),
        "disabled": False,
        "created_at": "2025-02-09 09:35:23",
        "processed_by": "kaxm23"
    }
}

# FastAPI app
app = FastAPI()

@app.post("/token",
          response_model=Token,
          description="Get access token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """
    Login endpoint to get access token.
    
    Args:
        form_data: OAuth2 password request form
        
    Returns:
        Token: Access token response
        
    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(
        fake_users_db,
        form_data.username,
        form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_at="2025-02-09 09:35:23",
        processed_by="kaxm23"
    )

@app.get("/users/me/",
         response_model=User,
         description="Get current user")
async def read_users_me(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current user endpoint.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Current user information
    """
    return current_user

@app.post("/users/",
          response_model=User,
          description="Create new user")
async def create_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    full_name: Optional[str] = None
) -> User:
    """
    Create new user endpoint.
    
    Args:
        username: Username
        password: Password
        email: Optional email
        full_name: Optional full name
        
    Returns:
        User: Created user information
        
    Raises:
        HTTPException: If user already exists
    """
    if username in fake_users_db:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(password)
    user_dict = {
        "username": username,
        "hashed_password": hashed_password,
        "email": email,
        "full_name": full_name,
        "disabled": False,
        "created_at": "2025-02-09 09:35:23",
        "processed_by": "kaxm23"
    }
    
    fake_users_db[username] = user_dict
    
    return User(**user_dict)