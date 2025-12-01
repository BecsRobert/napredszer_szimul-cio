#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D texture1;
uniform vec3 lightPos;

void main() {
    vec3 color = texture(texture1, TexCoord).rgb;
    
    vec3 ambient = 0.1 * color;
    
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;
    
    FragColor = vec4(ambient + diffuse, 1.0);
}