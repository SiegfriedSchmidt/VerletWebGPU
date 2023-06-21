@group(0) @binding(0) var<uniform> resolution: vec2f;
@group(0) @binding(1) var<uniform> time: f32;

fn pallete(t: f32) -> vec3f {
    let a = vec3f(0.5, 0.5, 0.5);
    let b = vec3f(0.5, 0.5, 0.5);
    let c = vec3f(1.0, 1.0, 1.0);
    let d = vec3f(0.263, 0.416, 0.557);
    return a + b * cos(6.28318 * ( c * t + d));
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  var uv = vec2f(pos.x, resolution.y - pos.y) / resolution * 2 - 1;
  var uv0 = uv;
  var finalColor = vec3f(0.0);

  for (var i = 0; i < 4; i++) {
      uv = fract(uv * 8) - 0.5;
      var d = length(uv) * exp(-length(uv0));

      var col = pallete(length(uv0) + f32(i)*.4 + time * .4);

      d = sin(d * 8 + time) / 8;
      d = abs(d);
      d = pow(0.01 / d, 1.2);

      finalColor += col * d;
  }

  return vec4f(finalColor, 1);
}