use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::thread;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec3 {
    e: [f64; 3],
}

type Color = Vec3;
type Point3 = Vec3;

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { e: [x, y, z] }
    }

    fn rand() -> Vec3 {
        let mut rng = thread_rng();
        Vec3 {
            e: [
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
                rng.gen_range(0.0, 1.0),
            ],
        }
    }

    fn rand_range(min: f64, max: f64) -> Vec3 {
        let mut rng = thread_rng();
        Vec3 {
            e: [
                rng.gen_range(min, max),
                rng.gen_range(min, max),
                rng.gen_range(min, max),
            ],
        }
    }

    fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.e[0].abs() < s && self.e[1].abs() < s && self.e[2].abs() < s
    }

    fn x(&self) -> f64 {
        self.e[0]
    }

    fn y(&self) -> f64 {
        self.e[1]
    }

    fn z(&self) -> f64 {
        self.e[2]
    }

    fn length_squared(&self) -> f64 {
        self.e[0] * self.e[0] + self.e[1] * self.e[1] + self.e[2] * self.e[2]
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            e: [
                self.e[0] + rhs.e[0],
                self.e[1] + rhs.e[1],
                self.e[2] + rhs.e[2],
            ],
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            e: [
                self.e[0] - rhs.e[0],
                self.e[1] - rhs.e[1],
                self.e[2] - rhs.e[2],
            ],
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, t: f64) -> Self {
        Self {
            e: [self.e[0] * t, self.e[1] * t, self.e[2] * t],
        }
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, t: f64) -> Self {
        (1.0 / t) * self
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            e: [vec.e[0] * self, vec.e[1] * self, vec.e[2] * self],
        }
    }
}

impl std::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            e: [
                vec.e[0] * self.e[0],
                vec.e[1] * self.e[1],
                vec.e[2] * self.e[2],
            ],
        }
    }
}

fn dot(v1: Vec3, v2: Vec3) -> f64 {
    v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z()
}

fn cross(u: Vec3, v: Vec3) -> Vec3 {
    Vec3::new(
        u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0],
    )
}

struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn at(&self, t: f64) -> Vec3 {
        self.origin + t * self.direction
    }
}

fn unit_vector(v: Vec3) -> Vec3 {
    v / v.length()
}

#[derive(Clone, Copy)]
struct Camera {
    aspect_ratio: f64,
    viewport_height: f64,
    viewport_width: f64,

    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Vec3,

    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f64,
}

fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

impl Camera {
    fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = degrees_to_radians(vfov);
        let h = (theta / 2.0).tan();
        let viewport_height: f64 = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = unit_vector(lookfrom - lookat);
        let u = unit_vector(cross(vup, w));
        let v = cross(w, u);

        let origin = lookfrom;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;
        let lens_radius = aperture / 2.0;

        Camera {
            aspect_ratio,
            viewport_height,
            viewport_width,
            origin,
            horizontal,
            vertical,
            lower_left_corner,
            u,
            v,
            w,
            lens_radius,
        }
    }

    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x() + self.v * rd.y();

        Ray {
            origin: self.origin + offset,
            direction: self.lower_left_corner + s * self.horizontal + t * self.vertical
                - self.origin
                - offset,
        }
    }
}

struct HitRecord {
    p: Point3,
    normal: Vec3,
    t: f64,
    material: Arc<dyn Material>,
    front_face: bool,
}

impl HitRecord {
    fn new(
        ray: &Ray,
        p: Point3,
        outward_normal: Vec3,
        t: f64,
        material: Arc<dyn Material>,
    ) -> Self {
        let front_face = dot(ray.direction, outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -1.0 * outward_normal
        };

        HitRecord {
            p,
            normal,
            t,
            material,
            front_face,
        }
    }
}

trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

struct Sphere {
    center: Vec3,
    radius: f64,
    material: Arc<dyn Material>,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Arc<dyn Material>) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = dot(ray.direction, ray.direction);
        let b = dot(oc, ray.direction);
        let c = dot(oc, oc) - self.radius.powi(2);
        let discriminant = b.powi(2) - a * c;
        if discriminant > 0.0 {
            let sqrtd = discriminant.sqrt();
            let t = (-b - sqrtd) / a;
            if t < t_max && t > t_min {
                let p = ray.at(t);
                let normal = (p - self.center) / self.radius;
                return Some(HitRecord::new(ray, p, normal, t, self.material.clone()));
            }
            /*
             * I have to comment this out!
             *
            let t = (-b + sqrtd) / a;
            if t < t_max && t > t_min {
                let p = ray.at(t);
                let normal = (p - self.center) / self.radius;
                return Some(HitRecord::new(ray, p, normal, t, self.material.clone()));
            }
            */
        }
        None
    }
}

type World = Vec<Box<dyn Hittable>>;

impl Hittable for World {
    fn hit(&self, r: &Ray, _: f64, _: f64) -> Option<HitRecord> {
        let mut result: Option<HitRecord> = None;

        let mut closest = std::f64::MAX;
        for t in self {
            if let Some(hit_record) = t.hit(r, 0.0, closest) {
                closest = hit_record.t;
                result = Some(hit_record);
            }
        }
        result
    }
}

struct ScatterRecord {
    attenuation: Color,
    scattered: Ray,
}

trait Material: Send + Sync {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<ScatterRecord>;
}

struct Lambertian {
    albedo: Color,
}

impl Material for Lambertian {
    fn scatter(&self, _: &Ray, hit_record: &HitRecord) -> Option<ScatterRecord> {
        let mut scatter_direction = hit_record.normal + unit_vector(Vec3::rand());

        // Catch degenerate scatter direction
        if scatter_direction.near_zero() {
            scatter_direction = hit_record.normal;
        }
        Some(ScatterRecord {
            scattered: Ray {
                origin: hit_record.p,
                direction: scatter_direction,
            },
            attenuation: self.albedo,
        })
    }
}

struct Metal {
    albedo: Color,
    fuzz: f64,
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * dot(v, n) * n
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<ScatterRecord> {
        let mut reflected = reflect(unit_vector(ray.direction), hit_record.normal);
        if self.fuzz > 0.0 {
            reflected = reflected + self.fuzz * random_in_unit_sphere()
        };
        if dot(reflected, hit_record.normal) > 0.0 {
            let scattered = Ray {
                origin: hit_record.p,
                direction: reflected,
            };
            Some(ScatterRecord {
                scattered,
                attenuation: self.albedo,
            })
        } else {
            None
        }
    }
}

struct Dielectric {
    index_of_refraction: f64,
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Use Schlick's approximation for reflectance.
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = dot(-1.0 * uv, n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<ScatterRecord> {
        let attenuation = Color::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit_record.front_face {
            1.0 / self.index_of_refraction
        } else {
            self.index_of_refraction
        };

        let unit_direction = unit_vector(ray.direction);

        let cos_theta = dot(-1.0 * unit_direction, hit_record.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction = if cannot_refract
            || reflectance(cos_theta, refraction_ratio) > thread_rng().gen_range(0.0, 1.0)
        {
            reflect(unit_direction, hit_record.normal)
        } else {
            refract(unit_direction, hit_record.normal, refraction_ratio)
        };

        Some(ScatterRecord {
            scattered: Ray {
                origin: hit_record.p,
                direction,
            },
            attenuation,
        })
    }
}

fn random_scene() -> World {
    let mut world: World = Vec::new();

    let ground_material = Arc::new(Lambertian {
        albedo: Color::new(0.5, 0.5, 0.5),
    });
    world.push(Box::new(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    )));

    let mut rng = thread_rng();
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.gen_range(0.0, 1.0);
            let center = Point3::new(
                a as f64 + 0.9 * rng.gen_range(0.0, 1.0),
                0.2,
                b as f64 + 0.9 * rng.gen_range(0.0, 1.0),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Color::rand() * Color::rand();
                    let sphere_material = Arc::new(Lambertian { albedo });
                    world.push(Box::new(Sphere::new(center, 0.2, sphere_material)));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Color::rand_range(0.5, 1.0);
                    let fuzz = rng.gen_range(0.0, 0.5);
                    let sphere_material = Arc::new(Metal { albedo, fuzz });
                    world.push(Box::new(Sphere::new(center, 0.2, sphere_material)));
                } else {
                    // glass
                    let sphere_material = Arc::new(Dielectric {
                        index_of_refraction: 1.5,
                    });
                    world.push(Box::new(Sphere::new(center, 0.2, sphere_material)));
                }
            }
        }
    }

    let material1 = Arc::new(Dielectric {
        index_of_refraction: 1.5,
    });
    world.push(Box::new(Sphere::new(
        Point3::new(0.0, 1.0, 0.0),
        1.0,
        material1,
    )));

    let material2 = Arc::new(Lambertian {
        albedo: Color::new(0.4, 0.2, 0.1),
    });
    world.push(Box::new(Sphere::new(
        Point3::new(-4.0, 1.0, 0.0),
        1.0,
        material2,
    )));

    let material3 = Arc::new(Metal {
        albedo: Color::new(0.7, 0.6, 0.5),
        fuzz: 0.0,
    });
    world.push(Box::new(Sphere::new(
        Point3::new(4.0, 1.0, 0.0),
        1.0,
        material3,
    )));

    world
}

fn ray_color(ray: &Ray, world: &World, depth: u32) -> Color {
    if depth <= 0 {
        return Color::new(0.0, 0.0, 0.0);
    }
    match world.hit(ray, 0.001, std::f64::MAX) {
        Some(hit_record) => {
            if let Some(scatter_record) = hit_record.material.scatter(&ray, &hit_record) {
                scatter_record.attenuation * ray_color(&scatter_record.scattered, world, depth - 1)
            } else {
                Color::new(0.0, 0.0, 0.0)
            }
        }
        None => {
            let unit_direction = unit_vector(ray.direction);
            let t = 0.5 * (unit_direction.y() + 1.0);
            (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
        }
    }
}


fn main() {
    use chrono::prelude::*;
    use std::time::SystemTime;

    let now = SystemTime::now();
    println!("Start: {:?}", Local::now());
    generate();
    println!("Time: {:?}", now.elapsed().unwrap());
}

fn generate() {
    let aspect_ratio = 3.0 / 2.0;
    let image_width: u32 = 800;
    let image_height = ((image_width as f64) / aspect_ratio) as u32;

    let samples_per_pixel = 100;
    let max_depth = 50;

    let lookfrom = Point3::new(13.0, 2.0, 3.0);
    let lookat = Point3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let focus_dist = 10.0;
    let aperture = 0.1;
    let vfov = 20.0;

    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        vfov,
        aspect_ratio,
        aperture,
        focus_dist,
    );

    let world = Arc::new(random_scene());

    let mut file = File::create("output.ppm").unwrap();

    writeln!(file, "P3").unwrap();
    writeln!(file, "{} {}", image_width, image_height).unwrap();
    writeln!(file, "255").unwrap();

    generate_multi_thread(
        &mut file,
        image_height,
        image_width,
        samples_per_pixel,
        max_depth,
        &camera,
        world,
    );
}

fn generate_multi_thread(
    file: &mut File,
    image_height: u32,
    image_width: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    camera: &Camera,
    world: Arc<World>,
) {
    let v: Vec<u32> = (0..image_height).collect();
    let chunks = v.chunks(8);

    let mut handles = Vec::new();
    for chunk in chunks {
        let vec: Vec<u32> = Vec::from(chunk);
        let camera = camera.clone();
        let world = world.clone();
        let h = thread::spawn(move || {
            let mut pixels = Vec::new();
            for j in vec {
                let j = image_height - j - 1;
                for i in 0..image_width {
                    let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                    for _ in 0..samples_per_pixel {
                        let u = (i as f64 + thread_rng().gen_range(0.0, 1.0))
                            / (image_width - 1) as f64;
                        let v = (j as f64 + thread_rng().gen_range(0.0, 1.0))
                            / (image_height - 1) as f64;

                        let r = camera.get_ray(u, v);
                        pixel_color = pixel_color + ray_color(&r, &world, max_depth);
                    }
                    pixels.push(pixel_color);
                }
            }
            pixels
        });
        handles.push(h);
    }
    for h in handles {
        let pixels = h.join().unwrap();
        for pixel_color in pixels {
            write_color(file, pixel_color, samples_per_pixel);
        }
    }
}

fn generate_single_thread(
    file: &mut File,
    image_height: u32,
    image_width: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    camera: &Camera,
    world: Arc<World>,
) {
    for j in 0..image_height {
        let j = image_height - j - 1;
        for i in 0..image_width {
            let mut pixel_color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..samples_per_pixel {
                let u = (i as f64 + thread_rng().gen_range(0.0, 1.0)) / (image_width - 1) as f64;
                let v = (j as f64 + thread_rng().gen_range(0.0, 1.0)) / (image_height - 1) as f64;

                let r = camera.get_ray(u, v);
                pixel_color = pixel_color + ray_color(&r, &world, max_depth);
            }
            write_color(file, pixel_color, samples_per_pixel);
        }
    }
}

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = Vec3::rand_range(-1.0, 1.0);
        if p.length_squared() > 1.0 {
            continue;
        }
        return p;
    }
}

fn random_in_unit_disk() -> Vec3 {
    let mut rng = thread_rng();
    loop {
        let p = Vec3::new(rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0), 0.0);
        if p.length_squared() >= 1.0 {
            continue;
        }
        return p;
    }
}

fn random_in_hemisphere(normal: Vec3) -> Vec3 {
    let in_unit_sphere = random_in_unit_sphere();
    if dot(in_unit_sphere, normal) > 0.0 {
        in_unit_sphere
    } else {
        -1.0 * in_unit_sphere
    }
}

fn write_color(file: &mut File, color: Color, samples_per_pixel: u32) {
    let scale = 1.0 / samples_per_pixel as f64;
    let r = (color.x() * scale).sqrt();
    let g = (color.y() * scale).sqrt();
    let b = (color.z() * scale).sqrt();
    writeln!(
        file,
        "{} {} {}",
        (clamp(r, 0.0, 1.0) * 255.99) as u8,
        (clamp(g, 0.0, 1.0) * 255.99) as u8,
        (clamp(b, 0.0, 1.0) * 255.99) as u8
    )
    .unwrap();
}
