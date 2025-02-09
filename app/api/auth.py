from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional, Dict
import os

from app.models.user_auth import User, UserRole, UserStatus
from app.database.session import get_async_session
from app.core.config import Settings

router = APIRouter()
settings = Settings()

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UserCreate(BaseModel):
    """User creation request model."""
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain number')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Validate passwords match."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserResponse(BaseModel):
    """User response model."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    status: str
    created_at: str
    timestamp: str = "2025-02-09 09:41:18"
    processed_by: str = "kaxm23"

class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserResponse
    timestamp: str = "2025-02-09 09:41:18"
    processed_by: str = "kaxm23"

class TokenRefresh(BaseModel):
    """Token refresh request model."""
    refresh_token: str

async def create_access_token(data: dict,
                            expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
        "processed_by": "kaxm23"
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "processed_by": "kaxm23"
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session)
) -> User:
    """Get current user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "access":
            raise credentials_exception
        
        result = await session.execute(
            select(User).filter(User.username == username)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            raise credentials_exception
            
        return user
        
    except JWTError:
        raise credentials_exception

@router.post("/signup",
             response_model=UserResponse,
             status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_async_session)
) -> UserResponse:
    """
    User signup endpoint.
    
    Args:
        user_data: User creation data
        session: Database session
        
    Returns:
        UserResponse: Created user data
        
    Raises:
        HTTPException: If username/email exists or creation fails
    """
    # Check existing user
    result = await session.execute(
        select(User).filter(
            or_(
                User.username == user_data.username,
                User.email == user_data.email
            )
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    try:
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=UserRole.USER
        )
        
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            status=user.status.value,
            created_at=user.created_at.isoformat(),
            timestamp="2025-02-09 09:41:18",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

@router.post("/login",
             response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session)
) -> Token:
    """
    User login endpoint.
    
    Args:
        form_data: OAuth2 password request form
        session: Database session
        
    Returns:
        Token: Access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    # Get user
    result = await session.execute(
        select(User).filter(User.username == form_data.username)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.verify_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check account status
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status.value}"
        )
    
    # Check account lock
    if user.is_account_locked():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is temporarily locked"
        )
    
    try:
        # Create tokens
        access_token = await create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        refresh_token = await create_refresh_token(
            data={"sub": user.username}
        )
        
        # Record successful login
        user.record_login_attempt(success=True)
        await session.commit()
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                status=user.status.value,
                created_at=user.created_at.isoformat(),
                timestamp="2025-02-09 09:41:18",
                processed_by="kaxm23"
            ),
            timestamp="2025-02-09 09:41:18",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/refresh",
             response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    session: AsyncSession = Depends(get_async_session)
) -> Token:
    """
    Refresh token endpoint.
    
    Args:
        token_data: Refresh token data
        session: Database session
        
    Returns:
        Token: New access and refresh tokens
        
    Raises:
        HTTPException: If refresh fails
    """
    try:
        # Verify refresh token
        payload = jwt.decode(
            token_data.refresh_token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid refresh token"
            )
        
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid refresh token"
            )
        
        # Get user
        result = await session.execute(
            select(User).filter(User.username == username)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create new tokens
        access_token = await create_access_token(
            data={"sub": username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        refresh_token = await create_refresh_token(
            data={"sub": username}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                status=user.status.value,
                created_at=user.created_at.isoformat(),
                timestamp="2025-02-09 09:41:18",
                processed_by="kaxm23"
            ),
            timestamp="2025-02-09 09:41:18",
            processed_by="kaxm23"
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    User logout endpoint.
    
    Args:
        current_user: Current authenticated user
        session: Database session
    """
    return {"message": "Successfully logged out"}