# -*- coding: utf-8 -*-
"""
===================================
TickFlowFetcher - 付费数据源 (Priority 动态)
===================================

数据来源：TickFlow API (https://tickflow.com)
特点：专业金融数据服务，接口稳定、数据质量高
仓库：pip install tickflow

免费额度：
- 历史日 K 线数据（无需 API Key）
- 无实时行情

付费额度：
- 实时行情（需 API Key）
- 更高频数据

优先级策略（动态）：
- 配置了 TICKFLOW_API_KEY：priority=0（最高优先级）
- 未配置 API Key：priority=99（最低优先级，仅历史K线可用）

防封禁策略：
1. 使用 tenacity 实现指数退避重试
2. 熔断器机制：连续失败后自动冷却
3. API Key 未配置时优雅降级
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS
from .realtime_types import (
    UnifiedRealtimeQuote, RealtimeSource,
    get_realtime_circuit_breaker,
    safe_float, safe_int,
)


logger = logging.getLogger(__name__)


# === TickFlow SDK 可用性标志 ===
_TICKFLOW_AVAILABLE = False
_tickflow_import_error: Optional[str] = None

try:
    from tickflow import TickFlow
    _TICKFLOW_AVAILABLE = True
    logger.debug("TickFlow SDK 导入成功")
except ImportError as e:
    _tickflow_import_error = str(e)
    logger.warning(f"TickFlow SDK 未安装，TickFlowFetcher 将不可用。安装方法: pip install tickflow")


# === TickFlow 客户端缓存 ===
_tf_client: Optional[Any] = None


def _get_tf_client(api_key: Optional[str] = None) -> Optional[Any]:
    """
    获取或创建 TickFlow 客户端（懒加载单例）

    Args:
        api_key: API Key（可选，免费额度不需要）

    Returns:
        TickFlow 客户端实例，SDK 未安装时返回 None
    """
    global _tf_client

    if not _TICKFLOW_AVAILABLE:
        return None

    if _tf_client is not None:
        return _tf_client

    try:
        if api_key:
            _tf_client = TickFlow(api_key=api_key)
            logger.info("TickFlow 客户端初始化成功（付费模式）")
        else:
            _tf_client = TickFlow()
            logger.info("TickFlow 客户端初始化成功（免费模式）")
    except Exception as e:
        logger.error(f"TickFlow 客户端初始化失败: {e}")
        return None

    return _tf_client


def _is_etf_code(stock_code: str) -> bool:
    """
    判断代码是否为 ETF 基金

    ETF 代码规则：
    - 上交所 ETF: 51xxxx, 52xxxx, 56xxxx, 58xxxx
    - 深交所 ETF: 15xxxx, 16xxxx, 18xxxx

    Args:
        stock_code: 股票/基金代码

    Returns:
        True 表示是 ETF 代码，False 表示是普通股票代码
    """
    etf_prefixes = ('51', '52', '56', '58', '15', '16', '18')
    return stock_code.startswith(etf_prefixes) and len(stock_code) == 6


def _is_hk_code(stock_code: str) -> bool:
    """
    判断代码是否为港股

    港股代码规则：
    - 5位数字代码，如 '00700' (腾讯控股)
    - 可能带有前缀，如 'hk00700', 'hk1810'

    Args:
        stock_code: 股票代码

    Returns:
        True 表示是港股代码
    """
    code = stock_code.lower()
    if code.startswith('hk'):
        numeric_part = code[2:]
        return numeric_part.isdigit() and 1 <= len(numeric_part) <= 5
    return code.isdigit() and len(code) == 5


def _is_us_code(stock_code: str) -> bool:
    """
    判断代码是否为美股

    美股代码规则：
    - 1-5个大写字母，如 'AAPL' (苹果), 'TSLA' (特斯拉)

    Args:
        stock_code: 股票代码

    Returns:
        True 表示是美股代码
    """
    import re
    code = stock_code.strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', code))


def _convert_to_tickflow_symbol(stock_code: str) -> str:
    """
    将标准股票代码转换为 TickFlow 格式

    TickFlow 符号格式：
    - A 股（沪市）：600519 -> 600519.SH
    - A 股（深市）：000001 -> 000001.SZ
    - ETF（沪市）：510300 -> 510300.SH
    - ETF（深市）：159919 -> 159919.SZ
    - 港股：hk00700 -> 00700.HK
    - 美股：AAPL -> AAPL（保持不变）

    Args:
        stock_code: 标准股票代码

    Returns:
        TickFlow 格式的符号
    """
    code = stock_code.strip()

    # 已经包含后缀的情况（直接返回）
    if '.' in code:
        return code.upper()

    # 港股处理
    if _is_hk_code(code):
        numeric_part = code.lower().replace('hk', '').zfill(5)
        return f"{numeric_part}.HK"

    # 美股处理（TickFlow 直接使用字母代码）
    if _is_us_code(code):
        return code.upper()

    # A 股 / ETF 处理
    # 沪市：600xxx, 601xxx, 603xxx, 605xxx, 688xxx (科创板), 51xxxx, 52xxxx, 56xxxx, 58xxxx (ETF)
    sh_prefixes = ('600', '601', '603', '605', '688', '51', '52', '56', '58', '9')
    if code.startswith(sh_prefixes):
        return f"{code}.SH"
    else:
        # 深市：000xxx, 001xxx, 002xxx, 003xxx, 300xxx (创业板), 15xxxx, 16xxxx, 18xxxx (ETF)
        return f"{code}.SZ"


class TickFlowFetcher(BaseFetcher):
    """
    TickFlow 数据源实现

    优先级策略（动态）：
    - 配置了 TICKFLOW_API_KEY 且 SDK 可用：priority=0（最高优先级）
    - SDK 可用但无 API Key：priority=98（仅历史K线可用，低于免费数据源）
    - SDK 未安装：priority=99（不可用）

    数据来源：TickFlow API
    文档：https://tickflow.com/docs

    主要 API：
    - tf.klines.get(): 获取历史 K 线数据
    - tf.realtime.get(): 获取单只股票实时行情（需付费）
    - tf.quotes.get(): 获取多只股票实时行情（需付费）

    关键策略：
    - 动态优先级：根据 API Key 和 SDK 状态自动调整
    - 指数退避重试（最多3次）
    - SDK 未安装时优雅降级
    """

    name = "TickFlowFetcher"
    priority = 99  # 默认最低优先级，会在 __init__ 中根据配置动态调整

    def __init__(self):
        """
        初始化 TickFlowFetcher

        根据以下因素动态确定优先级：
        1. TickFlow SDK 是否已安装
        2. TICKFLOW_API_KEY 是否已配置
        """
        self._api_key: Optional[str] = None
        self._client: Optional[Any] = None

        # 动态优先级确定
        self.priority = self._determine_priority()

    def _determine_priority(self) -> int:
        """
        根据 SDK 可用性和 API Key 配置确定优先级

        策略：
        - SDK 可用 + API Key 配置：priority=0（最高）
        - SDK 可用 + 无 API Key：priority=98（仅历史K线，低于免费数据源）
        - SDK 不可用：priority=99（完全不可用）

        Returns:
            优先级数字（0=最高，数字越大优先级越低）
        """
        # Step 1: 检查 SDK 是否安装
        if not _TICKFLOW_AVAILABLE:
            logger.warning("TickFlow SDK 未安装，数据源不可用（priority=99）")
            logger.warning(f"  导入错误: {_tickflow_import_error}")
            logger.warning("  安装方法: pip install tickflow")
            return 99

        # Step 2: 检查 API Key 配置
        try:
            from src.config import get_config
            config = get_config()
            self._api_key = getattr(config, 'tickflow_api_key', None)
        except Exception:
            self._api_key = None

        if self._api_key:
            # 尝试初始化客户端
            self._client = _get_tf_client(self._api_key)
            if self._client is not None:
                logger.info("TickFlow SDK 可用且 API Key 已配置，优先级提升为最高 (Priority 0)")
                return 0
            else:
                logger.warning("TickFlow API Key 配置但客户端初始化失败 (priority=98)")
                return 98
        else:
            # 无 API Key，仅免费历史 K 线
            self._client = _get_tf_client()
            if self._client is not None:
                logger.info("TickFlow SDK 可用（免费模式，无实时行情），priority=98")
                return 98
            else:
                logger.warning("TickFlow 客户端初始化失败 (priority=99)")
                return 99

    def is_available(self) -> bool:
        """
        检查数据源是否可用

        Returns:
            True 表示可用，False 表示不可用
        """
        return self._client is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 TickFlow 获取原始数据

        使用 TickFlow klines.get() API 获取历史 K 线数据

        流程：
        1. 检查客户端是否可用
        2. 转换股票代码为 TickFlow 格式
        3. 调用 tf.klines.get() API
        4. 处理返回数据

        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'

        Returns:
            原始数据 DataFrame

        Raises:
            DataFetchError: 获取数据失败
        """
        # 检查 SDK 可用性
        if not _TICKFLOW_AVAILABLE:
            raise DataFetchError(f"TickFlow SDK 未安装: {_tickflow_import_error}")

        # 获取客户端
        client = self._client or _get_tf_client(self._api_key)
        if client is None:
            raise DataFetchError("TickFlow 客户端未初始化")

        # 转换代码格式
        symbol = _convert_to_tickflow_symbol(stock_code)

        logger.info(f"[API调用] tf.klines.get(symbol={symbol}, period=1d, "
                   f"start_date={start_date}, end_date={end_date})")

        try:
            import time as _time
            api_start = _time.time()

            # 调用 TickFlow klines API 获取日线数据
            df = client.klines.get(
                symbol=symbol,
                period="1d",
                start_date=start_date,
                end_date=end_date,
                as_dataframe=True,
            )

            api_elapsed = _time.time() - api_start

            # 记录返回数据摘要
            if df is not None and not df.empty:
                logger.info(f"[API返回] tf.klines.get 成功: 返回 {len(df)} 行数据, 耗时 {api_elapsed:.2f}s")
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                # 尝试获取日期范围
                date_col = self._find_date_column(df)
                if date_col:
                    logger.info(f"[API返回] 日期范围: {df[date_col].iloc[0]} ~ {df[date_col].iloc[-1]}")
                logger.debug(f"[API返回] 最新3条数据:\n{df.tail(3).to_string()}")
            else:
                logger.warning(f"[API返回] tf.klines.get 返回空数据, 耗时 {api_elapsed:.2f}s")

            return df

        except DataFetchError:
            raise

        except Exception as e:
            error_msg = str(e).lower()

            # 检测 API 限制
            if any(keyword in error_msg for keyword in ['limit', 'quota', '频率', 'rate', '限制', '权限']):
                logger.warning(f"TickFlow API 可能被限流: {e}")
                raise RateLimitError(f"TickFlow 可能被限流: {e}") from e

            raise DataFetchError(f"TickFlow 获取数据失败: {e}") from e

    @staticmethod
    def _find_date_column(df: pd.DataFrame) -> Optional[str]:
        """
        查找 DataFrame 中的日期列

        TickFlow 可能返回不同的日期列名：
        - datetime
        - date
        - trade_date
        - time

        Args:
            df: 原始数据 DataFrame

        Returns:
            日期列名，未找到返回 None
        """
        date_candidates = ['datetime', 'date', 'trade_date', 'time', 'timestamp']
        for col in date_candidates:
            if col in df.columns:
                return col
        return None

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 TickFlow 数据

        TickFlow klines API 可能返回的列名：
        datetime / date, open, high, low, close, volume, amount, pct_chg, turnover, ...

        映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # === 日期列处理 ===
        date_col = self._find_date_column(df)
        if date_col and date_col != 'date':
            df = df.rename(columns={date_col: 'date'})

        # === 标准列名映射 ===
        column_mapping = {
            # 日期列
            'datetime': 'date',
            'trade_date': 'date',
            'time': 'date',
            'timestamp': 'date',
            # 价格列
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            # 量价列
            'vol': 'volume',
            'volume': 'volume',
            'amount': 'amount',
            'turnover': 'amount',
            # 涨跌幅列
            'pct_chg': 'pct_chg',
            'change_pct': 'pct_chg',
            'pct_change': 'pct_chg',
            'change_percent': 'pct_chg',
        }

        df = df.rename(columns=column_mapping)

        # === 日期格式标准化 ===
        if 'date' in df.columns:
            # 尝试多种日期格式解析
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # 过滤无效日期
            df = df.dropna(subset=['date'])

        # === 成交量单位标准化 ===
        # TickFlow 的 volume 单位可能是手，转换为股（统一为股）
        if 'volume' in df.columns:
            # 如果成交量数值较小（< 1e6），可能单位是手，需要 * 100
            max_vol = df['volume'].max()
            if max_vol and max_vol < 100000:
                logger.debug(f"[TickFlow] 检测到成交量单位可能为手，自动转换为股 (max={max_vol})")
                df['volume'] = df['volume'] * 100

        # === 添加股票代码列 ===
        df['code'] = stock_code

        # === 只保留需要的列 ===
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情数据

        数据来源：
        - tf.quotes.get(): 批量获取多只股票实时行情（推荐）
        - tf.realtime.get(): 获取单只股票实时行情

        注意：实时行情需要 API Key（付费功能）

        Args:
            stock_code: 股票代码

        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        # 检查 SDK 可用性
        if not _TICKFLOW_AVAILABLE:
            logger.debug(f"[实时行情-TickFlow] SDK 未安装，跳过 {stock_code}")
            return None

        # 检查 API Key（实时行情需要付费）
        if not self._api_key:
            logger.debug(f"[实时行情-TickFlow] 未配置 API Key，实时行情不可用，跳过 {stock_code}")
            return None

        circuit_breaker = get_realtime_circuit_breaker()
        source_key = "tickflow"

        # 检查熔断器状态
        if not circuit_breaker.is_available(source_key):
            logger.warning(f"[熔断] 数据源 {source_key} 处于熔断状态，跳过")
            return None

        # 获取客户端
        client = self._client or _get_tf_client(self._api_key)
        if client is None:
            logger.warning(f"[实时行情-TickFlow] 客户端未初始化，跳过 {stock_code}")
            return None

        # 转换代码格式
        symbol = _convert_to_tickflow_symbol(stock_code)

        try:
            logger.info(f"[API调用] tf.quotes.get(symbols=[{symbol}]) 获取实时行情...")

            import time as _time
            api_start = _time.time()

            # 优先使用 quotes.get() 批量接口
            df = client.quotes.get(symbols=[symbol])

            api_elapsed = _time.time() - api_start

            if df is not None and not df.empty:
                logger.info(f"[API返回] tf.quotes.get 成功: 耗时 {api_elapsed:.2f}s")
                logger.info(f"[API返回] 列名: {list(df.columns)}")
                circuit_breaker.record_success(source_key)
            else:
                # quotes 接口失败时，尝试 realtime.get() 单股票接口
                logger.info(f"[API返回] tf.quotes.get 返回空数据，尝试 tf.realtime.get()...")

                df = client.realtime.get(symbol=symbol, as_dataframe=True)

                api_elapsed = _time.time() - api_start

                if df is not None and not df.empty:
                    logger.info(f"[API返回] tf.realtime.get 成功: 耗时 {api_elapsed:.2f}s")
                    logger.info(f"[API返回] 列名: {list(df.columns)}")
                    circuit_breaker.record_success(source_key)
                else:
                    logger.warning(f"[API返回] TickFlow 实时行情数据为空，跳过 {stock_code}")
                    return None

            # 解析实时行情数据
            # TickFlow 实时行情列名可能包括：symbol/name/code, price/latest_price, change_pct, volume, amount 等
            row = df.iloc[0]

            # 灵活匹配列名
            name = self._extract_field(row, ['name', 'stock_name', 'sec_name'], default='')
            price = safe_float(self._extract_field(row, ['price', 'latest_price', 'close', 'last']))
            change_pct = safe_float(self._extract_field(row, ['change_pct', 'pct_chg', 'change_percent', 'price_change_pct']))
            change_amount = safe_float(self._extract_field(row, ['change', 'change_amount', 'price_change']))
            volume = safe_int(self._extract_field(row, ['volume', 'vol', 'trade_volume']))
            amount = safe_float(self._extract_field(row, ['amount', 'turnover', 'trade_amount']))
            turnover_rate = safe_float(self._extract_field(row, ['turnover_rate', 'turnover_ratio', 'exchange_rate']))
            amplitude = safe_float(self._extract_field(row, ['amplitude', 'swing']))
            open_price = safe_float(self._extract_field(row, ['open', 'open_price', 'open_interest']))
            high = safe_float(self._extract_field(row, ['high', 'high_price', 'highest']))
            low = safe_float(self._extract_field(row, ['low', 'low_price', 'lowest']))

            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=str(name),
                source=RealtimeSource.FALLBACK,  # TickFlow 暂无专属 source enum
                price=price,
                change_pct=change_pct,
                change_amount=change_amount,
                volume=volume,
                amount=amount,
                turnover_rate=turnover_rate,
                amplitude=amplitude,
                open_price=open_price,
                high=high,
                low=low,
            )

            logger.info(f"[实时行情-TickFlow] {stock_code} {quote.name}: 价格={quote.price}, "
                       f"涨跌={quote.change_pct}%")
            return quote

        except Exception as e:
            logger.error(f"[API错误] 获取 {stock_code} 实时行情(TickFlow)失败: {e}")
            circuit_breaker.record_failure(source_key, str(e))
            return None

    @staticmethod
    def _extract_field(row: pd.Series, candidates: List[str], default: Any = None) -> Any:
        """
        从 DataFrame 行中按候选列名提取字段

        TickFlow 不同版本/接口可能返回不同列名，
        此方法按优先级尝试匹配。

        Args:
            row: DataFrame 的一行
            candidates: 候选列名列表（按优先级排列）
            default: 默认值

        Returns:
            找到的值，未找到返回默认值
        """
        for col in candidates:
            if col in row.index:
                val = row[col]
                if val is not None and str(val).strip() != '' and str(val) != 'nan':
                    return val
        return default


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    fetcher = TickFlowFetcher()

    print("=" * 60)
    print(f"TickFlowFetcher 初始化状态")
    print(f"  名称: {fetcher.name}")
    print(f"  优先级: {fetcher.priority}")
    print(f"  可用: {fetcher.is_available()}")
    print("=" * 60)

    if not fetcher.is_available():
        print("TickFlowFetcher 不可用，请检查:")
        print(f"  1. SDK 是否安装: pip install tickflow")
        print(f"  2. API Key 是否配置: TICKFLOW_API_KEY")
    else:
        # 测试普通 A 股
        print("\n" + "=" * 60)
        print("测试普通 A 股数据获取 (TickFlow)")
        print("=" * 60)
        try:
            df = fetcher.get_daily_data('600519')  # 茅台
            print(f"[A股] 获取成功，共 {len(df)} 条数据")
            print(df.tail())
        except Exception as e:
            print(f"[A股] 获取失败: {e}")

        # 测试 ETF
        print("\n" + "=" * 60)
        print("测试 ETF 基金数据获取 (TickFlow)")
        print("=" * 60)
        try:
            df = fetcher.get_daily_data('510300')  # 沪深300 ETF
            print(f"[ETF] 获取成功，共 {len(df)} 条数据")
            print(df.tail())
        except Exception as e:
            print(f"[ETF] 获取失败: {e}")

        # 测试港股
        print("\n" + "=" * 60)
        print("测试港股数据获取 (TickFlow)")
        print("=" * 60)
        try:
            df = fetcher.get_daily_data('hk00700')  # 腾讯控股
            print(f"[港股] 获取成功，共 {len(df)} 条数据")
            print(df.tail())
        except Exception as e:
            print(f"[港股] 获取失败: {e}")

        # 测试实时行情（需要 API Key）
        print("\n" + "=" * 60)
        print("测试实时行情获取 (TickFlow)")
        print("=" * 60)
        try:
            quote = fetcher.get_realtime_quote('600519')
            if quote:
                print(f"[实时行情] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
            else:
                print("[实时行情] 未获取到数据（可能需要 API Key）")
        except Exception as e:
            print(f"[实时行情] 获取失败: {e}")
