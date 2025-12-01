#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in mat4 aInstanceMatrix; 

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 view;
uniform mat4 projection;

void main() {
    TexCoord = aTexCoord;
    
    Normal = mat3(transpose(inverse(aInstanceMatrix))) * aNormal;
    
    vec4 pos = aInstanceMatrix * vec4(aPos, 1.0);
    FragPos = vec3(pos);
    
    gl_Position = projection * view * pos;
}