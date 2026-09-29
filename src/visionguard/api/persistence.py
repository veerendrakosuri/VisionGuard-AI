"""SQLAlchemy persistence compatible with PostgreSQL and local SQLite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

from sqlalchemy import DateTime, Float, String, create_engine, delete, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from visionguard.api.models import AnalyticsSummary, InspectionResult, ModelInfo


class Base(DeclarativeBase):  # type: ignore[misc]
    pass


class InspectionRecord(Base):
    __tablename__ = "inspections"

    inspection_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    decision: Mapped[str] = mapped_column(String(4), index=True)
    anomaly_score: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_source: Mapped[str] = mapped_column(String(32))
    inference_time_ms: Mapped[float] = mapped_column(Float)
    heatmap_url: Mapped[str] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64), index=True)
    backbone: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class InspectionRepository:
    def __init__(self, database_url: str) -> None:
        if database_url.startswith("sqlite:///"):
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, pool_pre_ping=True)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def save(self, result: InspectionResult) -> None:
        with Session(self.engine) as session:
            session.merge(
                InspectionRecord(
                    inspection_id=result.inspection_id,
                    filename=result.filename,
                    decision=result.decision,
                    anomaly_score=result.anomaly_score,
                    threshold=result.threshold,
                    threshold_source=result.threshold_source,
                    inference_time_ms=result.inference_time_ms,
                    heatmap_url=result.heatmap_url,
                    model_name=result.model.name,
                    category=result.model.category,
                    backbone=result.model.backbone,
                    created_at=result.created_at,
                )
            )
            session.commit()

    def get(self, inspection_id: str) -> InspectionResult | None:
        with Session(self.engine) as session:
            record = session.get(InspectionRecord, inspection_id)
            return None if record is None else self._to_result(record)

    def list(
        self,
        limit: int,
        offset: int,
        decision: Literal["PASS", "FAIL"] | None = None,
    ) -> tuple[list[InspectionResult], int]:
        condition = InspectionRecord.decision == decision if decision else None
        query = select(InspectionRecord).order_by(InspectionRecord.created_at.desc())
        count_query = select(func.count()).select_from(InspectionRecord)
        if condition is not None:
            query = query.where(condition)
            count_query = count_query.where(condition)
        with Session(self.engine) as session:
            records = session.scalars(query.limit(limit).offset(offset)).all()
            total = int(session.scalar(count_query) or 0)
        return [self._to_result(record) for record in records], total

    def analytics(self) -> AnalyticsSummary:
        with Session(self.engine) as session:
            total = int(session.scalar(select(func.count()).select_from(InspectionRecord)) or 0)
            fails = int(
                session.scalar(
                    select(func.count()).where(InspectionRecord.decision == "FAIL")
                )
                or 0
            )
            averages = session.execute(
                select(
                    func.avg(InspectionRecord.anomaly_score),
                    func.avg(InspectionRecord.inference_time_ms),
                )
            ).one()
        return AnalyticsSummary(
            total_inspections=total,
            pass_count=total - fails,
            fail_count=fails,
            fail_rate=round(fails / total, 4) if total else 0.0,
            average_anomaly_score=round(float(averages[0] or 0), 4),
            average_inference_time_ms=round(float(averages[1] or 0), 2),
        )

    def delete_expired(self, retention_days: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        with Session(self.engine) as session:
            result = session.execute(
                delete(InspectionRecord).where(InspectionRecord.created_at < cutoff)
            )
            session.commit()
            return int(result.rowcount or 0)

    @staticmethod
    def _to_result(record: InspectionRecord) -> InspectionResult:
        return InspectionResult(
            inspection_id=record.inspection_id,
            filename=record.filename,
            decision=cast(Literal["PASS", "FAIL"], record.decision),
            anomaly_score=record.anomaly_score,
            threshold=record.threshold,
            threshold_source=cast(
                Literal["configured", "model_provisional"], record.threshold_source
            ),
            inference_time_ms=record.inference_time_ms,
            heatmap_url=record.heatmap_url,
            model=ModelInfo(
                name=record.model_name,
                category=record.category,
                backbone=record.backbone,
            ),
            created_at=record.created_at,
        )
