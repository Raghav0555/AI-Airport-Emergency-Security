pipeline {
    agent any

    environment {
        DOCKERHUB_USER = 'raghav1089'
        IMAGE_NAME     = 'raghav1089/aies-app'
        IMAGE_TAG      = "${env.BUILD_NUMBER}"
    }

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'ls -la'
            }
        }

        stage('Build & Test') {
            steps {
                sh 'python3 --version'
                sh 'python3 -m compileall -q app src simulation evaluation experiments'
                echo 'Syntax check passed for all Python modules'
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG -t $IMAGE_NAME:latest .'
                sh 'docker images $IMAGE_NAME'
            }
        }

        stage('Push Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
                    sh 'echo "$DH_PASS" | docker login -u "$DH_USER" --password-stdin'
                    sh 'docker push $IMAGE_NAME:$IMAGE_TAG'
                    sh 'docker push $IMAGE_NAME:latest'
                    sh 'docker logout'
                }
            }
        }

        stage('Minikube Deployment') {
            steps {
                sh 'kubectl apply -f k8s/'
                sh 'kubectl set image deployment/aies-app aies-app=$IMAGE_NAME:$IMAGE_TAG'
                sh 'kubectl rollout status deployment/aies-app --timeout=240s'
            }
        }

        stage('Verification') {
            steps {
                sh 'kubectl get pods -o wide'
                sh 'kubectl get deployments'
                sh 'kubectl get svc'
                sh 'curl -fsS --retry 12 --retry-delay 5 --retry-connrefused http://$(minikube ip):30080/_stcore/health'
            }
        }
    }

    post {
        success {
            echo "CI/CD pipeline completed successfully for build #${env.BUILD_NUMBER}"
            echo 'AIES Airport Simulator is deployed on Kubernetes (Minikube) at NodePort 30080'
        }
        failure {
            echo "Pipeline failed for build #${env.BUILD_NUMBER}"
        }
        always {
            sh 'docker image prune -f || true'
        }
    }
}