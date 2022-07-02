var can;
var gl;

var this_frame;
var delta_time;
var last_frame;

const cos = Math.cos;
const sin = Math.sin;
const PI = Math.PI;

const tex_size = 1024;

const world_size_x = 32;
const world_size_y = 32;
const world_size_z = 32;
const world_size_w = 32;

const KEY_COMMAND_XYZ = '1';
const KEY_COMMAND_XYW = '2';
const KEY_COMMAND_XZW = '3';
const KEY_COMMAND_YZW = '4';

const KEY_COMMAND_TWIRL_CW = 't';
const KEY_COMMAND_TWIRL_CCW = 'g';
const KEY_COMMAND_STRAFE_FORWARD = 'w';
const KEY_COMMAND_STRAFE_BACKWARD = 's';
const KEY_COMMAND_STRAFE_LEFT = 'a';
const KEY_COMMAND_STRAFE_RIGHT = 'd';
const KEY_COMMAND_STRAFE_UP = ' ';
const KEY_COMMAND_STRAFE_DOWN = 'Shift';
const KEY_COMMAND_STRAFE_WINT = 'r';
const KEY_COMMAND_STRAFE_ZANT = 'f';

function loadShader(gl,source,type){
	let shader = gl.createShader(type);
	gl.shaderSource(shader,source);
	gl.compileShader(shader);
	console.log(gl.getShaderInfoLog(shader));
	return shader;
}

function initShaderProgram(gl,vs,fs){
	let vertex = loadShader(gl,vs,gl.VERTEX_SHADER);
	let fragment = loadShader(gl,fs,gl.FRAGMENT_SHADER);
	let program = gl.createProgram();
	gl.attachShader(program,vertex);
	gl.attachShader(program,fragment);
	gl.linkProgram(program);
	console.log(gl.getProgramInfoLog(program));
	return program;
}

function initShader(gl,vs,fs){
	let program = initShaderProgram(gl,vs,fs);
	let uniformLocations = {};
	
	let numUniforms = gl.getProgramParameter(program,gl.ACTIVE_UNIFORMS);
	for(let i = 0; i < numUniforms; i++){
		let name = gl.getActiveUniform(program,i).name;
		console.log(name);
		uniformLocations[name] = gl.getUniformLocation(program,name);
	}
	
	
	
	return {
		program: program,
		uniformLocations: uniformLocations
	};

	
};

let data_texture;
let blocks_texture;
let main_shader;

var vsSource = `#version 300 es
out vec4 v_glp;

void main(){
	v_glp = vec4(-1+4*(gl_VertexID%2), -1+4*(gl_VertexID/2), 0., 1.);
	gl_Position = v_glp;
	
	return;
}
`;

var fsSource = `#version 300 es
precision highp float;
precision highp int;

in vec4 v_glp;

uniform mat4 u_mv_orr;
uniform vec4 u_mv_trn;
uniform mat4 u_proj;
uniform vec2 u_res;

uniform sampler2D u_tex;
uniform highp usampler2D u_datatex;

struct Hit4 {
	vec4 N;
	float t;
};

int voxelmap(ivec4 P){
	
	const int Wx = `+world_size_x+`;
	const int Wy = `+world_size_y+`;
	const int Wz = `+world_size_z+`;
	const int Ww = `+world_size_w+`;
	
	if(P.x < 0 || P.x >= Wx) return -1;
	if(P.y < 0 || P.y >= Wy) return -1;
	if(P.z < 0 || P.z >= Wz) return -1;
	if(P.w < 0 || P.w >= Ww) return -1;
	//return 1.;
	int ind = P.x+Wx*(P.y+Wy*(P.z+Wz*P.w));
	
	uint samp = texelFetch(u_datatex, ivec2((ind>>2)%`+tex_size+`, (ind>>2)/`+tex_size+`), 0)[(ind>>2)%4];
	
	return int((samp>>uint(8*(ind%4)))&255u);
}

Hit4 ray_voxelmap(vec4 ro, vec4 rd, out int block_type, out vec4 block_rel){
    const float epsilon = 0.00001;
    float depth = 0.0;
    float stepsize = 1.0;
    for(int i = 0; i < 256; i++){
        vec4 p = fract(ro + (depth)*rd);
        vec4 o = 1.0 - max(vec4(0.0), sign(-rd));
        float q1 = min(
            -(p.w-o.w)/rd.w,
            min(
            -(p.x-o.x)/rd.x,
            min(
            -(p.y-o.y)/rd.y,
            -(p.z-o.z)/rd.z
            )));
        int voxelsamp = voxelmap(ivec4(ro + rd*(depth+q1+epsilon)));
        if(voxelsamp > 0){
            vec4 N = floor(ro + rd*depth)-floor(ro + rd*(depth+q1+epsilon));
            block_rel = fract(ro + rd*(depth+q1+epsilon));
            block_type = voxelsamp;
            return Hit4(N, depth+q1-epsilon);
        }
        depth += q1+epsilon;
    }
    return Hit4(vec4(0.0),0.0);
}



vec3 hue(float x){
    x = fract(x)*6.0;
    return vec3(
    min(1.0,max(max(0.0,min(1.0,2.0-x)),x-4.0)),
    max(0.0,min(min(x,1.0),4.0-x)),
    min(1.0,min(max(0.0,x-2.0),6.0-x))
    );
}

vec3 block_texcoord_privatefunc(vec4 rel, vec4 N){
	const mat3x4 mX = mat3x4(
	 0., 0., 1., 0.,
	 0., 1., 0., 0.,
	 0., 0., 0.,-1.
	);
	const mat3x4 mZ = mat3x4(
	 1., 0., 0., 0.,
	 0., 1., 0., 0.,
	 0., 0., 0.,-1.
	);
	const mat3x4 mW = mat3x4(
	 1., 0., 0., 0.,
	 0., 1., 0., 0.,
	 0., 0.,-1., 0.
	);
	
	const mat3x4 mYp = mat3x4(
	 0., 0.,-1., 0.,
	-1., 0., 0., 0.,
	 0., 0., 0.,-1.
	);
	const mat3x4 mYn = mat3x4(
	 1., 0., 0., 0.,
	 0., 0., 1., 0.,
	 0., 0., 0.,-1.
	);
	
	vec4 aN = abs(N);
	float aNmaxc = max(aN.x, max(aN.y, max(aN.z, aN.w)));
	
	vec3 uvs;
	
	if(aN.x == aNmaxc){
		return transpose(mX)*rel;
	}
	if(aN.z == aNmaxc){
		return transpose(mZ)*rel;
	}
	if(aN.w == aNmaxc){
		return transpose(mW)*rel;
	}
	
	if(aN.y == aNmaxc){
		return transpose(N.y > 0.0 ? mYp : mYn)*rel;
	}
	
	
}
vec3 block_texcoord(vec4 rel, vec4 N){
	return mod(block_texcoord_privatefunc(rel,N), vec3(1.));
}

float id_uv_offset(int id, vec4 N){
	int side = 0;
	if(N.y < 0.5) side++;
	if(N.y < -0.5) side++;
	
	float o = float(2*side);
	
	if(id == 1){
		if(side == 2) return 144.;
		return 16.*(1.+o);
	}
	if(id == 2){
		return 80.+16.*o;
	}
	if(id == 3){
		return 176.+16.*o;
	}
	if(id == 4){
		return 304.+16.*o;
	}
	if(id == 5){
		return 400.+16.*o;
	}
	if(id == 7){
		return 496.+16.*o;
	}
	if(id == 8){
		return 592.+16.*o;
	}
	if(id == 9){
		return 688.+16.*o;
	}
	
	
	
	if(id == 10){
		if(side == 2) return 912.;
		return 784.+16.*o;
	}
	if(id == 11){
		return 848.+16.*o;
	}
	if(id == 12){
		return 1136.+16.*o;
	}
	if(id == 13){
		return 944.+16.*o;
	}
	if(id == 14){
		return 1040.+16.*o;
	}
	
	if(id == 6){
		return 272.;
	}
}

out vec4 fragColor;

float ao_vertex(ivec4 bpos, vec4 N, ivec4 vertex){
	float ao = 0.;
	
	for(int i = 0; i < 16; i++){
		int x = i&1;
		int y = (i>>1)&1;
		int z = (i>>2)&1;
		int w = (i>>3)&1;
		if(N.x != 0.) x = 0;
		if(N.y != 0.) y = 0;
		if(N.z != 0.) z = 0;
		if(N.w != 0.) w = 0;
		ao += voxelmap(bpos+vertex+ivec4(x,y,z,w)) > 0 ? 1. : 0.;
	}
	
	return ao;
}

float get_ao(vec4 );

void main(){
	vec4 clip = vec4(-1.+2.*gl_FragCoord.xy/u_res, 0., 1.);
	
	vec4 unproj_rd = inverse(u_proj)*clip;
	vec4 rd = transpose(u_mv_orr)*vec4(normalize(unproj_rd.xyz/unproj_rd.w),0.);
	vec4 ro = transpose(u_mv_orr)*-u_mv_trn;
	
	int btype;
	vec4 brel;
	
	Hit4 hit = ray_voxelmap(ro, rd+vec4(0.00001), btype, brel);
	
	vec3 tcoord =  block_texcoord(brel, hit.N);
	vec2 tuv = vec2( floor(16.*tcoord.x), floor(16.*tcoord.y) + 16.*floor(16.*tcoord.z));
	tuv.x += id_uv_offset(btype, hit.N);
	
	fragColor = texelFetch(u_tex, ivec2(tuv.x, 255.-tuv.y), 0);
	
	vec4 hitpos = ro + hit.t*rd;
	ivec4 hitb = ivec4(hitpos);
	
	float ao = 0.;
	
	ao += ao_vertex(hitb, hit.N, ivec4(0,0,0,0));
	
	//fragColor.xyz *= ao;
	
	if(!(hit.t > 0.0)){
		fragColor = vec4(0.,0.,0.,1.);
	}
}
`;

function camera_align(key){
  switch(key){
    case KEY_COMMAND_XYZ:
      camera_orr = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
      break;

    case KEY_COMMAND_XYW:
      camera_orr = [0,0,0,1, 0,1,0,0, 1,0,0,0, 0,0,1,0];
      break;

    case KEY_COMMAND_YZW:
      camera_orr = [0,0,-1,0,0,1,0,0,0,0,0,-1,1,0,0,0];
      break;
  }

  camera_mode = "up_aligned";
  return;
}

function camera_report(){
  console.log("Camera orientation is now:", camera_orr);
  console.log(".. with determinant:", glMatrix.mat4.determinant(camera_orr));
  console.log(".. and mode:", camera_mode);
}

function camera_toggle(){
	if(camera_mode == "up_aligned"){
		let wint_hor = [camera_orr[12],camera_orr[14],camera_orr[15]];
		let right_hor = [camera_orr[0],camera_orr[2],camera_orr[3]];
		let forward_hor = glMatrix.vec3.create();
		glMatrix.vec3.cross(forward_hor, right_hor, wint_hor);
		
		camera_orr[8] = -forward_hor[0];
		camera_orr[9] = 0.;
		camera_orr[10] = -forward_hor[1];
		camera_orr[11] = -forward_hor[2];
		
		camera_orr[4] = -wint_hor[0];
		camera_orr[5] = 0.;
		camera_orr[6] = -wint_hor[1];
		camera_orr[7] = -wint_hor[2];
		
		camera_orr[12] = 0.;
		camera_orr[13] = 1.;
		camera_orr[14] = 0.;
		camera_orr[15] = 0.;
		
		camera_mode = "horizontal";
		return;
	}
	if(camera_mode == "horizontal"){
		let backward_hor = [camera_orr[8],camera_orr[10],camera_orr[11]];
		let right_hor = [camera_orr[0],camera_orr[2],camera_orr[3]];
		
		camera_orr[8] = backward_hor[0];
		camera_orr[9] = 0.;
		camera_orr[10] = backward_hor[1];
		camera_orr[11] = backward_hor[2];
		
		let wint_hor = glMatrix.vec3.create();
		glMatrix.vec3.cross(wint_hor, right_hor, backward_hor);
		
		camera_orr[12] = wint_hor[0];
		camera_orr[13] = 0.;
		camera_orr[14] = wint_hor[1];
		camera_orr[15] = wint_hor[2];
		
		camera_orr[4] = 0.;
		camera_orr[5] = 1.;
		camera_orr[6] = 0.;
		camera_orr[7] = 0.;
		
		camera_mode = "up_aligned";
		return;
	}
}

function orthonormalize4(out, a){
    let tmp1;
    let tmp2;
    let tmp3;
    let tmp4;
    let tmp5;
    let tmp6;
    let tmp7;
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    tmp4 = out[0]*a[4]+out[1]*a[5]+out[2]*a[6]+out[3]*a[7];
    tmp5 = out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3];
    tmp1 = tmp4/tmp5;
    out[4] = a[4] - out[0]*tmp1;
    out[5] = a[5] - out[1]*tmp1;
    out[6] = a[6] - out[2]*tmp1;
    out[7] = a[7] - out[3]*tmp1;
    tmp4 = out[0]*a[8]+out[1]*a[9]+out[2]*a[10]+out[3]*a[11];
    tmp1 = tmp4/tmp5;
    tmp4 = out[4]*a[8]+out[5]*a[9]+out[6]*a[10]+out[7]*a[11];
    tmp6 = out[4]*out[4]+out[5]*out[5]+out[6]*out[6]+out[7]*out[7];
    tmp2 = tmp4/tmp6;
    out[8] = a[8] - out[0]*tmp1 - out[4]*tmp2;
    out[9] = a[9] - out[1]*tmp1 - out[5]*tmp2;
    out[10] = a[10] - out[2]*tmp1 - out[6]*tmp2;
    out[11] = a[11] - out[3]*tmp1 - out[7]*tmp2;
    tmp4 = out[0]*a[12]+out[1]*a[13]+out[2]*a[14]+out[3]*a[15];
    tmp1 = tmp4/tmp5;
    tmp4 = out[4]*a[12]+out[5]*a[13]+out[6]*a[14]+out[7]*a[15];
    tmp2 = tmp4/tmp6;
    tmp4 = out[8]*a[12]+out[9]*a[13]+out[10]*a[14]+out[11]*a[15];
    tmp7 = out[8]*out[8]+out[9]*out[9]+out[10]*out[10]+out[11]*out[11];
    tmp3 = tmp4/tmp7;
    out[12] = a[12] - out[0]*tmp1 - out[4]*tmp2 - out[8]*tmp3;
    out[13] = a[13] - out[1]*tmp1 - out[5]*tmp2 - out[9]*tmp3;
    out[14] = a[14] - out[2]*tmp1 - out[6]*tmp2 - out[10]*tmp3;
    out[15] = a[15] - out[3]*tmp1 - out[7]*tmp2 - out[11]*tmp3;
    tmp4 = out[12]*out[12]+out[13]*out[13]+out[14]*out[14]+out[15]*out[15];
    tmp1 = Math.sqrt(tmp4);
    out[12] /= tmp1; out[13] /= tmp1; out[14] /= tmp1; out[15] /= tmp1;
    tmp1 = Math.sqrt(tmp5);
    out[0] /= tmp1; out[1] /= tmp1; out[2] /= tmp1; out[3] /= tmp1;
    tmp1 = Math.sqrt(tmp6);
    out[4] /= tmp1; out[5] /= tmp1; out[6] /= tmp1; out[7] /= tmp1;
    tmp1 = Math.sqrt(tmp7);
    out[8] /= tmp1; out[9] /= tmp1; out[10] /= tmp1; out[11] /= tmp1;
}


function camera_frame(){
	let wint_hor = [camera_orr[12],camera_orr[14],camera_orr[15]];
	let right_hor = [camera_orr[0],camera_orr[2],camera_orr[3]];
	let forward_hor = glMatrix.vec3.create();
	glMatrix.vec3.cross(forward_hor, right_hor, wint_hor);
	
	
	let dt = 0.001*delta_time;
	
	dt = Math.min(1, dt);
	
	let spd = 70;
	let disp = dt*spd;
	
	if(camera_mode == "up_aligned"){
		let tmp = glMatrix.mat3.create();
		let tmpq = glMatrix.quat.create();
		glMatrix.quat.setAxisAngle(tmpq, right_hor, -to_scroll*0.2);
		glMatrix.mat3.fromQuat(tmp, tmpq);
		let scroll_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(scroll_mat,
		tmp[3*0+0],0.0,tmp[3*0+1],tmp[3*0+2],
		0.0,       1.0,0.0,       0.0,
		tmp[3*1+0],0.0,tmp[3*1+1],tmp[3*1+2],
		tmp[3*2+0],0.0,tmp[3*2+1],tmp[3*2+2]
		);
		glMatrix.mat4.multiply(camera_orr, scroll_mat, camera_orr);
		to_scroll *= 0.8;
		
		glMatrix.quat.setAxisAngle(tmpq, forward_hor, to_twirl*0.2);
		glMatrix.mat3.fromQuat(tmp, tmpq);
		let twirl_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(twirl_mat,
		tmp[3*0+0],0.0,tmp[3*0+1],tmp[3*0+2],
		0.0,       1.0,0.0,       0.0,
		tmp[3*1+0],0.0,tmp[3*1+1],tmp[3*1+2],
		tmp[3*2+0],0.0,tmp[3*2+1],tmp[3*2+2]
		);
		glMatrix.mat4.multiply(camera_orr, twirl_mat, camera_orr);
		to_twirl *= 0.8;
		
		camera_orr[1] = 0.;
		camera_orr[13] = 0.;
		
		
		if(keys_held[KEY_COMMAND_STRAFE_WINT]){
			camera_vel[0] += disp*wint_hor[0];
			camera_vel[2] += disp*wint_hor[1];
			camera_vel[3] += disp*wint_hor[2];
		}
		if(keys_held[KEY_COMMAND_STRAFE_ZANT]){
			camera_vel[0] += disp*-wint_hor[0];
			camera_vel[2] += disp*-wint_hor[1];
			camera_vel[3] += disp*-wint_hor[2];
		}
		
		
		if(keys_held[KEY_COMMAND_STRAFE_UP]){
			camera_vel[1] += disp;
		}
		if(keys_held[KEY_COMMAND_STRAFE_DOWN]){
			camera_vel[1] -= disp;
		}
	}
	
	if(camera_mode == "horizontal"){
		
		let tilt_diff = to_scroll*0.2;
		
		let pitch_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(pitch_mat,
		1.0,0.0,0.0,0.0,
		0.0,1.0,0.0,0.0,
		0.0,0.0,cos(tilt_diff),sin(tilt_diff),
		0.0,0.0,-sin(tilt_diff),cos(tilt_diff)
		);
		glMatrix.mat4.multiply(camera_orr, camera_orr, pitch_mat);
		
		to_scroll *= 0.8;
		
		
		let twirl_diff = to_twirl*0.2;
		
		let twirl_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(twirl_mat,
		cos(twirl_diff),-sin(twirl_diff),0.,0.,
		sin(twirl_diff),cos(twirl_diff),0.,0.,
		0.,0.,1.,0,
		0.,0.,0.,1.
		);
		glMatrix.mat4.multiply(camera_orr, camera_orr, twirl_mat);
		
		to_twirl *= 0.8;
		
		
		/*
		camera_orr[1] = 0.;
		camera_orr[9] = 0.;
		camera_orr[12] = 0.;
		camera_orr[13] = 1.;
		camera_orr[14] = 0.;
		camera_orr[15] = 0.;
		*/
		
		forward_hor[0] = -camera_orr[8];
		forward_hor[1] = -camera_orr[10];
		forward_hor[2] = -camera_orr[11];
		
		
		if(keys_held[KEY_COMMAND_STRAFE_UP]){
			camera_vel[0] += disp*camera_orr[4];
			camera_vel[1] += disp*camera_orr[5];
			camera_vel[2] += disp*camera_orr[6];
			camera_vel[3] += disp*camera_orr[7];
		}
		if(keys_held[KEY_COMMAND_STRAFE_DOWN]){
			camera_vel[0] += disp*-camera_orr[4];
			camera_vel[1] += disp*-camera_orr[5];
			camera_vel[2] += disp*-camera_orr[6];
			camera_vel[3] += disp*-camera_orr[7];
		}
		
		if(keys_held[KEY_COMMAND_STRAFE_WINT]){
			camera_vel[1] += 0.2*disp;
		}
		if(keys_held[KEY_COMMAND_STRAFE_ZANT]){
			camera_vel[1] -= 0.2*disp;
		}
	}
	
	
	if(keys_held[KEY_COMMAND_STRAFE_FORWARD]){
		camera_vel[0] += disp*forward_hor[0];
		camera_vel[2] += disp*forward_hor[1];
		camera_vel[3] += disp*forward_hor[2];
	}
	if(keys_held[KEY_COMMAND_STRAFE_BACKWARD]){
		camera_vel[0] += disp*-forward_hor[0];
		camera_vel[2] += disp*-forward_hor[1];
		camera_vel[3] += disp*-forward_hor[2];
	}
	if(keys_held[KEY_COMMAND_STRAFE_RIGHT]){
		camera_vel[0] += disp*right_hor[0];
		camera_vel[2] += disp*right_hor[1];
		camera_vel[3] += disp*right_hor[2];
	}
	if(keys_held[KEY_COMMAND_STRAFE_LEFT]){
		camera_vel[0] += disp*-right_hor[0];
		camera_vel[2] += disp*-right_hor[1];
		camera_vel[3] += disp*-right_hor[2];
	}
	
	
	
	
	
	
	if(keys_held[KEY_COMMAND_TWIRL_CW]) to_twirl += dt;
	if(keys_held[KEY_COMMAND_TWIRL_CCW]) to_twirl -= dt;
	
	
	orthonormalize4(camera_orr, camera_orr);
	
	let det = glMatrix.mat4.determinant(camera_orr);
	if(Math.abs(det) < 0.001){
		glMatrix.mat4.identity(camera_orr);
		det = 1.;
		console.log("degenerate matrix");
	}
	if(det < 0){
		camera_orr[12] *= -1;
		camera_orr[13] *= -1;
		camera_orr[14] *= -1;
		camera_orr[15] *= -1;
		console.log("camera determinant was negative");
	}
	
	glMatrix.vec4.add(camera_pos, camera_pos, camera_vel.map(x=>x*dt));
	
	glMatrix.vec4.scale(camera_vel, camera_vel, Math.exp(-10*dt));
}
function camera_mouse(delta_x, delta_y){
	let yaw_diff = mouse_sensitivity*delta_x;
	let pitch_diff = mouse_sensitivity*delta_y;
	
	let wint_hor = [camera_orr[12],camera_orr[14],camera_orr[15]];
	let right_hor = [camera_orr[0],camera_orr[2],camera_orr[3]];
	let forward_hor = glMatrix.vec3.create();
	glMatrix.vec3.cross(forward_hor, wint_hor, right_hor);
		
	if(camera_mode == "up_aligned"){
		let tmp = glMatrix.mat3.create();
		let tmpq = glMatrix.quat.create();
		glMatrix.quat.setAxisAngle(tmpq, wint_hor, yaw_diff);
		glMatrix.mat3.fromQuat(tmp, tmpq);
		let yaw_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(yaw_mat,
		tmp[3*0+0],0.0,tmp[3*0+1],tmp[3*0+2],
		0.0,       1.0,0.0,       0.0,
		tmp[3*1+0],0.0,tmp[3*1+1],tmp[3*1+2],
		tmp[3*2+0],0.0,tmp[3*2+1],tmp[3*2+2]
		);
		glMatrix.mat4.multiply(camera_orr, yaw_mat, camera_orr);
		
		
		let pitch_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(pitch_mat,
		1.0,0.0,0.0,0.0,
		0.0,cos(pitch_diff),-sin(pitch_diff),0.0,
		0.0,sin(pitch_diff),cos(pitch_diff),0.0,
		0.0,0.0,0.0,1.0
		);
		let tmp2 = glMatrix.mat4.create();
		glMatrix.mat4.multiply(tmp2, camera_orr, pitch_mat);
		if(tmp2[5] > 0.) camera_orr = tmp2;
		
		
	}
	if(camera_mode == "horizontal"){
		let yaw_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(yaw_mat,
		cos(yaw_diff),0.0,sin(yaw_diff),0.0,
		0.0,1.0,0.0,0.0,
		-sin(yaw_diff),0.0,cos(yaw_diff),0.0,
		0.0,0.0,0.0,1.0
		);
		
		
		let pitch_mat = glMatrix.mat4.create();
		glMatrix.mat4.set(pitch_mat,
		1.0,0.0,0.0,0.0,
		0.0,cos(pitch_diff),-sin(pitch_diff),0.0,
		0.0,sin(pitch_diff),cos(pitch_diff),0.0,
		0.0,0.0,0.0,1.0
		);
		
		
		glMatrix.mat4.multiply(camera_orr, camera_orr, yaw_mat);
		glMatrix.mat4.multiply(camera_orr, camera_orr, pitch_mat);
		
	}
}

function frame(){
	
	
	
	this_frame = performance.now();
	delta_time = this_frame-last_frame;
	last_frame = this_frame;
	
	
	
	
	camera_frame();
	
	
	
	
	
	//requestAnimationFrame(frame);
	setTimeout(frame, 1000/30);
	
	gl.clearColor(0.,0.,0.,1.);
	gl.clear(gl.COLOR_BUFFER_BIT);
	
	
	
	
	let mv_orr = glMatrix.mat4.create();
	let mv_trn = glMatrix.vec4.create();
	
	mv_trn[0] = -.5;
	mv_trn[1] = -.5;
	mv_trn[2] = -.5;
	mv_trn[3] = -.5;
	
	
	let ang1 = 0.001*performance.now();
	let ang2 = 0.0017*performance.now();
	let ang3 = 0.0025*performance.now();
	let ang4 = 0.003*performance.now();
	
	let tmp = new Float32Array([
	cos(ang1),0.,0.,sin(ang1),
	0.,1.,0.,0.,
	0.,0.,1.,0.,
	-sin(ang1),0.,0.,cos(ang1)
	]);
	glMatrix.mat4.multiply(mv_orr, tmp, mv_orr);
	
	tmp = new Float32Array([
	1.,0.,0.,0.,
	0.,cos(ang4),sin(ang4),0.,
	0.,-sin(ang4),cos(ang4),0.,
	0.,0.,0.,1.
	]);
	glMatrix.mat4.multiply(mv_orr, tmp, mv_orr);
	
	tmp = new Float32Array([
	1.,0.,0.,0.,
	0.,cos(ang2),0.,sin(ang2),
	0.,0.,1.,0.,
	0.,-sin(ang2),0.,cos(ang2)
	]);
	glMatrix.mat4.multiply(mv_orr, tmp, mv_orr);
	
	
	tmp = new Float32Array([
	1.,0.,0.,0.,
	0.,1.,0.,0.,
	0.,0.,cos(ang3),sin(ang3),
	0.,0.,-sin(ang3),cos(ang3)
	]);
	glMatrix.mat4.multiply(mv_orr, tmp, mv_orr);
	
	
	
	
	
	
	
	
	
	glMatrix.mat4.transpose(mv_orr, camera_orr); 
	
	mv_trn = camera_pos.map(x=>-x);
	
	
	glMatrix.vec4.transformMat4(mv_trn, mv_trn, mv_orr);
	
	//mv_trn[2] -= 2.0;
	mv_trn[3] += 0.00001;
	
	
	
	let persp = glMatrix.mat4.create();
	glMatrix.mat4.perspective(persp, Math.PI/2, gl.canvas.width/gl.canvas.height, 0.0001, Infinity);
	
	
	
	gl.useProgram(main_shader.program);
	
	gl.uniform2f(main_shader.uniformLocations["u_res"], gl.canvas.width, gl.canvas.height);
	gl.uniform1i(main_shader.uniformLocations["u_tex"], 0);
	gl.uniform1i(main_shader.uniformLocations["u_datatex"], 1);
	
	gl.uniformMatrix4fv(main_shader.uniformLocations["u_proj"], false, persp);
	gl.uniformMatrix4fv(main_shader.uniformLocations["u_mv_orr"], false, mv_orr);
	gl.uniform4f(main_shader.uniformLocations["u_mv_trn"], mv_trn[0], mv_trn[1], mv_trn[2], mv_trn[3]);
	
	gl.drawArrays(gl.TRIANGLES, 0, 3);
	
	
	/*
		
		ffmpeg -f rawvideo -pix_fmt rgb24 -video_size WxH -r 60 -i vid -vf vflip output.mp4
		
		
		 * */
	
	if(recording == true){
		gl.readPixels(0,0,video_width,video_height,gl.RGBA,gl.UNSIGNED_BYTE,video_curframe);
		for(let i = 0; i < video_width*video_height; i++){ //won't work with gl.RGB
			video_buffer[video_width*video_height*3*video_frame + 3*i+0] = video_curframe[4*i+0];
			video_buffer[video_width*video_height*3*video_frame + 3*i+1] = video_curframe[4*i+1];
			video_buffer[video_width*video_height*3*video_frame + 3*i+2] = video_curframe[4*i+2];
		}
		video_frame++;
		if(video_frame == video_framecount){
			downloadBlob(video_buffer, "vid", 'application/octet-stream');
			recording = false;
		}
	}
	
	render_frame++;
	
	let coords_str = "";
	coords_str += "X: "+(camera_pos[0].toFixed(2).padStart(5));
	coords_str += " Z: "+(camera_pos[2].toFixed(2).padStart(5));
	coords_str += " W: "+(camera_pos[3].toFixed(2).padStart(5));
	coords_str += " Y: "+(camera_pos[1].toFixed(2).padStart(5));
	
	let facing_str = "";
	facing_str += "X: "+((-camera_orr[8]).toFixed(1).padStart(4));
	facing_str += "   Z: "+((-camera_orr[10]).toFixed(1).padStart(4));
	facing_str += "   W: "+((-camera_orr[11]).toFixed(1).padStart(4));
	facing_str += "   Y: "+((-camera_orr[9]).toFixed(1).padStart(4));
	
	document.getElementById("momo").innerHTML = "<tt>"+((video_framecount-video_frame)/video_fps).toFixed(2) +" seconds left<br>"+coords_str+"<br>"+facing_str+"</tt>";
	
	
	
	window.sessionStorage.setItem("camera", JSON.stringify({pos:camera_pos,orr:camera_orr,mode:camera_mode}));
}



let camera_pos = glMatrix.vec4.create();
let camera_vel = glMatrix.vec4.create();
camera_pos[3] = 0.5;

let to_scroll = 0;
let to_twirl = 0;

let camera_orr = glMatrix.mat4.create();
let camera_mode = "up_aligned";
let mouse_sensitivity = 0.01;
let scroll_sensitivity = 0.001;

let keys_held = [];


//camera_mode = "horizontal";
//camera_orr = new Float32Array([1,0,0,0,0,0,0,-1,0,0,1,0,0,1,0,0]);


let world_blocks = new Uint8Array(world_size_x*world_size_y*world_size_z*world_size_w);

function in_bbb(x,y,z,w,x1,z1,w1,y1,x2,z2,w2,y2){
	if(x < x1) return false;
	if(x > x2) return false;
	if(y < y1) return false;
	if(y > y2) return false;
	if(z < z1) return false;
	if(z > z2) return false;
	if(w < w1) return false;
	if(w > w2) return false;
	return true;
}
function set_block(b,x,y,z,w){
	if(x < 0 || x >= world_size_x) return null;
	if(y < 0 || y >= world_size_y) return null;
	if(z < 0 || z >= world_size_z) return null;
	if(w < 0 || w >= world_size_w) return null;

	world_blocks[x+world_size_x*(y+world_size_y*(z+world_size_z*w))] = b;
}
function fill4h(b,x1,z1,w1,y1,x2,z2,w2,y2){
	for(let x = x1; x <= x2; x++){
		for(let y = y1; y <= y2; y++){
			for(let z = z1; z <= z2; z++){
				for(let w = w1; w <= w2; w++){
					set_block(b,x,y,z,w);
				}
			}
		}
	}
}

for(let W = 0; W < world_size_w; W++){
	for(let Z = 0; Z < world_size_z; Z++){
		for(let Y = 0; Y < world_size_y; Y++){
			for(let X = 0; X < world_size_x; X++){
				let i = X+world_size_x*(Y+world_size_y*(Z+world_size_z*W));
				
				//let h = Math.sin(X/5)*Math.sin(Z/5)*Math.sin(W/5)*8+20;
				
				world_blocks[i] = 0;
				if(Y == 0) world_blocks[i] = 1;
				
				let ch = X&Z&W&1;
				
				if(in_bbb(X,Y,Z,W,3,3,3,1,17,15,13,4)){
					if(!in_bbb(X,Y,Z,W,3+1,3+1,3+1,1,17-1,15-1,13-1,4)){
						world_blocks[i] = 3*ch;
					}
				}
			}
		}
	}
}

fill4h(3,3-1,3-1,3-1,5,17+1,15+1,13+1,5);
fill4h(3,3+2,3+2,3+2,1,17-2,15-2,13-2,1);


fill4h(4,16,16,16,1,25,26,27,6);
fill4h(0,17,17,17,1,24,25,26,6);
fill4h(3,16,16,16,6,25,26,27,6);

fill4h(3,16,17,17,1,16,21,21,4);
fill4h(0,16,18,18,1,16,20,20,3);

fill4h(3,9,14,9,0,11,20,11,0);
fill4h(3,9,18,12,0,11,20,20,0);
fill4h(3,12,18,18,0,16,20,20,0);


for(let i = 1; i < 15; i++){
	set_block(i,i,3,1,1);
}







world_blocks = loaded_world;


let recording = false;
const video_fps = 30;
const video_framecount = video_fps*60;
const video_width = 640;
const video_height = 480;

let render_frame = 0;
let video_frame = 0;
let video_curframe = new Uint8Array(video_width*video_height*4);
let video_buffer = new Uint8Array(video_width*video_height*3*video_framecount);
const downloadURL = (data, fileName) => {
  const a = document.createElement('a')
  a.href = data
  a.download = fileName
  document.body.appendChild(a)
  a.style.display = 'none'
  a.click()
  a.remove()
}

const downloadBlob = (data, fileName, mimeType) => {

  const blob = new Blob([data], {
    type: mimeType
  })

  const url = window.URL.createObjectURL(blob)

  downloadURL(url, fileName)

  setTimeout(() => window.URL.revokeObjectURL(url), 1000)
}






function init(){
	can = document.getElementById("canvas");
	gl = can.getContext("webgl2");
	
	
	
	try {
		let cam_json = JSON.parse(window.sessionStorage.getItem("camera"));
		console.log(cam_json);
		for(let i = 0; i < 16; i++){
			camera_orr[i] = parseFloat(cam_json.orr[(i).toString()]);
		}
		for(let i = 0; i < 4; i++){
			camera_pos[i] = parseFloat(cam_json.pos[(i).toString()]);
		}
		camera_mode = cam_json.mode;
	} catch(err){
		console.log(err);
	}
	
	blocks_texture = gl.createTexture();
	//https://webgl2fundamentals.org/webgl/lessons/webgl-3d-textures.html
	gl.activeTexture(gl.TEXTURE0 + 0);

	// bind to the TEXTURE_2D bind point of texture unit 0
	gl.bindTexture(gl.TEXTURE_2D, blocks_texture);

	// Fill the texture with a 1x1 blue pixel.
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
	new Uint8Array([0, 0, 255, 255]));
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

	// Asynchronously load an image
	var image = new Image();
	image.src = "tiles.png";
	image.addEventListener('load', function() {
	// Now that the image has loaded make copy it to the texture.
	gl.activeTexture(gl.TEXTURE0 + 0);
	gl.bindTexture(gl.TEXTURE_2D, blocks_texture);
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
	});
	
	
	//gl.enable(gl.BLEND);
	//gl.blendFunc(gl.ONE, gl.ONE);
	gl.enable(gl.DEPTH_TEST);
	
	window.addEventListener("keydown",(e)=>{
		keys_held[e.key] = true;
    switch(e.key)
    { case KEY_COMMAND_XZW: // toggle back and forth whether or not Y axis is vertical
        camera_toggle();
        // camera_report();
        break;

      case KEY_COMMAND_XYZ: // align some other horizontal hyperplane
      case KEY_COMMAND_XYW:
      case KEY_COMMAND_YZW:
        camera_align(e.key);
        // camera_report();
        break;

    }
	});
	window.addEventListener("keyup",(e)=>{
		keys_held[e.key] = false;
	});
	
	window.addEventListener("wheel",(e)=>{
		to_scroll += scroll_sensitivity*e.deltaY;
	});
	
	can.addEventListener("click",()=>{
		can.requestPointerLock();
	});
	can.addEventListener("mousemove",(e)=>{
	
		if(!(document.pointerLockElement === can || document.mozPointerLockElement === can)) return;
		
		
		
		camera_mouse(e.movementX, e.movementY);
		
		
	
	});
	
	
	
	
	data_texture = gl.createTexture();
	
	
	gl.activeTexture(gl.TEXTURE0+1);
	gl.bindTexture(gl.TEXTURE_2D, data_texture);
	
	
	let data = new Uint32Array(4 * (tex_size**2));
	
	for(let i = 0; i < data.length; i++){
		
		
		
		data[i] = (world_blocks[4*Math.floor(i/4)+0]) + (world_blocks[4*Math.floor(i/4)+1]<<8) + (world_blocks[4*Math.floor(i/4)+2]<<16) + (world_blocks[4*Math.floor(i/4)+3]<<24);
	}
	
	gl.texImage2D(
	gl.TEXTURE_2D,
	0,
	gl.RGBA32UI,
	tex_size,
	tex_size,
	0,
	gl.RGBA_INTEGER,
	gl.UNSIGNED_INT,
	new Uint32Array(data.length)
	);
	
	
	let stopwatch = performance.now();
	gl.texSubImage2D(
	gl.TEXTURE_2D,
	0,
	0,
	0,
	tex_size,
	tex_size,
	gl.RGBA_INTEGER,
	gl.UNSIGNED_INT,
	data
	);
	console.log("tex upload time: "+(performance.now()-stopwatch));
	
	
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	
	
	main_shader = initShader(gl, vsSource, fsSource);
	
	
	this_frame = performance.now();
	last_frame = performance.now();
	
	frame();
	
}
window.addEventListener("load", init);

/*
setTimeout(()=>{
	camera_pos[2] = 5;
	camera_pos[1] = 22;
}, 100);
*/






