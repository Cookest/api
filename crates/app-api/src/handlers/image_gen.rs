//! Image generation handlers — proxy to the image-gen Python service.
//!
//! All routes require JWT. The image-gen internal token is added server-side.
//!
//! Routes:
//!   POST /api/image-gen/recipes/{id}/steps/batch  — trigger generation for all steps
//!   POST /api/image-gen/recipes/{id}/steps/{n}    — trigger generation for one step
//!   POST /api/image-gen/recipes/{id}/hero         — trigger hero image generation
//!   GET  /api/image-gen/jobs/{job_id}             — poll job status

use actix_web::{web, HttpResponse};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use cookest_shared::errors::AppError;

// ── Client ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ImageGenClient {
    pub client: Arc<Client>,
    pub base_url: String,
    pub token: Option<String>,
}

impl ImageGenClient {
    pub fn new(base_url: String, token: Option<String>) -> Self {
        Self {
            client: Arc::new(
                Client::builder()
                    .timeout(std::time::Duration::from_secs(300)) // generation can take time
                    .build()
                    .expect("reqwest client"),
            ),
            base_url,
            token,
        }
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let mut req = self.client.post(&url);
        if let Some(tok) = &self.token {
            req = req.bearer_auth(tok);
        }
        req
    }

    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let mut req = self.client.get(&url);
        if let Some(tok) = &self.token {
            req = req.bearer_auth(tok);
        }
        req
    }
}

// ── Request / Response types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BatchStepRequest {
    pub recipe_name: String,
    pub cuisine: Option<String>,
    pub steps: Vec<StepInfo>,
}

#[derive(Debug, Deserialize)]
pub struct StepInfo {
    pub step_index: usize,
    pub step_description: String,
}

#[derive(Debug, Deserialize)]
pub struct SingleStepRequest {
    pub recipe_name: String,
    pub step_description: String,
    pub total_steps: usize,
    pub cuisine: Option<String>,
    pub seed: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct HeroRequest {
    pub recipe_name: String,
    pub description: Option<String>,
    pub cuisine: Option<String>,
    pub category: Option<String>,
    pub seed: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ImageGenBatchPayload {
    recipe_id: i64,
    recipe_name: String,
    cuisine: Option<String>,
    steps: Vec<ImageGenStepItem>,
}

#[derive(Debug, Serialize)]
struct ImageGenStepItem {
    step_index: usize,
    step_description: String,
}

#[derive(Debug, Serialize)]
struct ImageGenStepPayload {
    recipe_id: i64,
    recipe_name: String,
    step_index: usize,
    total_steps: usize,
    step_description: String,
    cuisine: Option<String>,
    seed: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ImageGenHeroPayload {
    recipe_id: i64,
    recipe_name: String,
    description: Option<String>,
    cuisine: Option<String>,
    category: Option<String>,
    seed: Option<i64>,
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// POST /api/image-gen/recipes/{id}/steps/batch
/// Trigger image generation for all recipe steps at once.
pub async fn batch_generate_steps(
    img: web::Data<ImageGenClient>,
    path: web::Path<i64>,
    body: web::Json<BatchStepRequest>,
) -> Result<HttpResponse, AppError> {
    let recipe_id = path.into_inner();
    let payload = ImageGenBatchPayload {
        recipe_id,
        recipe_name: body.recipe_name.clone(),
        cuisine: body.cuisine.clone(),
        steps: body
            .steps
            .iter()
            .map(|s| ImageGenStepItem {
                step_index: s.step_index,
                step_description: s.step_description.clone(),
            })
            .collect(),
    };

    let resp = img
        .post("/generate/batch")
        .json(&payload)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let status = resp.status();
    let body = resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    if status.is_success() {
        Ok(HttpResponse::Ok().json(body))
    } else {
        Ok(HttpResponse::BadGateway().json(body))
    }
}

/// POST /api/image-gen/recipes/{id}/steps/{n}
/// Trigger image generation for a single step.
pub async fn generate_step(
    img: web::Data<ImageGenClient>,
    path: web::Path<(i64, usize)>,
    body: web::Json<SingleStepRequest>,
) -> Result<HttpResponse, AppError> {
    let (recipe_id, step_n) = path.into_inner();
    let payload = ImageGenStepPayload {
        recipe_id,
        recipe_name: body.recipe_name.clone(),
        step_index: step_n,
        total_steps: body.total_steps,
        step_description: body.step_description.clone(),
        cuisine: body.cuisine.clone(),
        seed: body.seed,
    };

    proxy_post(&img, "/generate/step", &payload).await
}

/// POST /api/image-gen/recipes/{id}/hero
/// Trigger hero image generation for a recipe.
pub async fn generate_hero(
    img: web::Data<ImageGenClient>,
    path: web::Path<i64>,
    body: web::Json<HeroRequest>,
) -> Result<HttpResponse, AppError> {
    let recipe_id = path.into_inner();
    let payload = ImageGenHeroPayload {
        recipe_id,
        recipe_name: body.recipe_name.clone(),
        description: body.description.clone(),
        cuisine: body.cuisine.clone(),
        category: body.category.clone(),
        seed: body.seed,
    };

    proxy_post(&img, "/generate/hero", &payload).await
}

/// GET /api/image-gen/jobs/{job_id}
/// Poll a generation job.
pub async fn get_job_status(
    img: web::Data<ImageGenClient>,
    path: web::Path<String>,
) -> Result<HttpResponse, AppError> {
    let job_id = path.into_inner();
    let resp = img
        .get(&format!("/jobs/{}", job_id))
        .send()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let status = resp.status();
    let body = resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    if status.is_success() {
        Ok(HttpResponse::Ok().json(body))
    } else {
        Ok(HttpResponse::BadGateway().json(body))
    }
}

/// GET /api/image-gen/health
pub async fn health(img: web::Data<ImageGenClient>) -> Result<HttpResponse, AppError> {
    let resp = img
        .get("/health")
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("image-gen unreachable: {}", e)))?;
    let body = resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(HttpResponse::Ok().json(body))
}

// ── Helper ────────────────────────────────────────────────────────────────────

async fn proxy_post<T: Serialize>(
    img: &ImageGenClient,
    path: &str,
    payload: &T,
) -> Result<HttpResponse, AppError> {
    let resp = img
        .post(path)
        .json(payload)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    let status = resp.status();
    let body = resp
        .json::<serde_json::Value>()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    if status.is_success() {
        Ok(HttpResponse::Ok().json(body))
    } else {
        Ok(HttpResponse::BadGateway().json(body))
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn configure_image_gen(cfg_routes: &mut web::ServiceConfig) {
    cfg_routes
        .route(
            "/api/image-gen/recipes/{id}/steps/batch",
            web::post().to(batch_generate_steps),
        )
        .route(
            "/api/image-gen/recipes/{id}/steps/{n}",
            web::post().to(generate_step),
        )
        .route(
            "/api/image-gen/recipes/{id}/hero",
            web::post().to(generate_hero),
        )
        .route(
            "/api/image-gen/jobs/{job_id}",
            web::get().to(get_job_status),
        )
        .route("/api/image-gen/health", web::get().to(health));
}
